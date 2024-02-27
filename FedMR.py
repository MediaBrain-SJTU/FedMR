from load_data import *
import numpy as np
import torch
import argparse
import random
from models import *
import os
import logging
import datetime
from partition_data import *
import torch.optim as optim
from utils import *
import json
from torch import distributed as dist
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class MR_loss(nn.Module):

    def __init__(self,):
        super(MR_loss, self).__init__()
        self.eps = 1e-8
        self.reject_threshold = 1

    def forward(self, x, y):
        _, C = x.shape
        loss = 0.0
        uniq_l, uniq_c = y.unique(return_counts=True)
        n_count = 0
        for i, label in enumerate(uniq_l):
            if uniq_c[i] <= self.reject_threshold:
                continue
            x_label = x[y==label, :]
            x_label = x_label - x_label.mean(dim=0, keepdim=True)
            x_label = x_label / torch.sqrt(self.eps + x_label.var(dim=0, keepdim=True))

            N = x_label.shape[0]
            corr_mat = torch.matmul(x_label.t(), x_label)
            loss += (off_diagonal(corr_mat).pow(2)).mean()
            n_count += N

        if n_count == 0:
            return 0
        else:
            loss = loss / n_count
            return loss

def set_seed(args):
    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_label', type=int, default=2,
                        help='num of label of each client')
    parser.add_argument('--temerature', type=float, default=0.5,
                        help='temperature of moon')
    parser.add_argument('--mu', type=float, default=0.01,
                        help='param of fedprox or moon')
    parser.add_argument('--skip', type=int, default=1,
                        help='num of skipping')
    parser.add_argument('--num_label', type=int, default=1,
                        help='num of classes per class')
    parser.add_argument('--logdir', type=str, required=False, default="/mnt/cache/fanziqing/SG/logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="/mnt/petrelfs/fanziqing/models/", help='Model directory path')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--train_batchsize', type=int, default=64, help='batchsize for training')
    parser.add_argument('--test_batchsize', type=int, default=64, help='batchsize for testing')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--model', type=str, default='simple-cnn', help='neural network used in training')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--party_per_round', type=int, default=10, help='how many clients are sampled in each round')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication roun')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_model_round', type=int, default=None,
                        help='how many rounds have executed for the loaded model')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    args = parser.parse_args()
    return args

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, optimizer,reg):

    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    if optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=reg)
    elif optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=reg,
                               amsgrad=True)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=reg)
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_deco=MR_loss().cuda()
    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector_1 = []
        epoch_loss_collector_2 = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _, h, out = net(x)
            loss_cls = criterion(out, target)
            loss_deco=criterion_deco(h,target)*args.mu
            loss=loss_cls+loss_deco
            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector_1.append(loss_cls.item())
            epoch_loss_collector_2.append(loss_deco.item())

        epoch_loss_1 = sum(epoch_loss_collector_1) / len(epoch_loss_collector_1)
        epoch_loss_2 = sum(epoch_loss_collector_2) / len(epoch_loss_collector_2)
        logger.info('Epoch: %d Loss_cls: %f Loss_deco: %f' % (epoch, epoch_loss_1,epoch_loss_2))

    test_acc,  _ = compute_accuracy(net, test_dataloader)

    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')

    logger.info(' ** Training complete **')
    return 0, test_acc


def local_train_net(nets, selected, args, train_dl_set, test_dl):

    for net_id, net in nets.items():
        if net_id in selected:
            logger.info("Training network %s" % (str(net_id)))
            trainacc, testacc = train_net(net_id, net, train_dl_set[net_id], test_dl, args.epochs, args.lr,
                                          args.optimizer, args.reg)
            logger.info("net %d final test acc %f" % (net_id, testacc))

    return nets


if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))

    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    set_seed(args)

    logger.info("Partitioning data")
    if args.dataset in ["cifar10","cifar100","tiny-imagenet","mnist","fmnist","SVHN"]:
        if args.dataset in ["cifar10","mnist","fmnist","SVHN"]:
            n_class=10
        elif args.dataset=="cifar100":
            n_class=100
        elif args.dataset == "tiny-imagenet":
            n_class=200
        _, _, _, _, net_dataidx_map, _ = partition_class(args.dataset,
                                                         args.n_parties,
                                                         n_class,
                                                         args.n_label)

        train_dl_local_set, test_dl, train_ds_local_set, test_ds = init_dataloader(args,net_dataidx_map)
    elif args.dataset in ["femnist_by_writer", "synthetic", "shakespeare", "PACS", "officehome","ISIC"]:
        train_dl_local_set, test_dl, train_ds_local_set, test_ds = init_dataloader(args)

    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    for i in range(args.comm_round):
        party_list_rounds.append(random.sample(party_list, args.party_per_round))

    train_dl = None
    logger.info("Initializing nets")
    nets = init_nets(args,args.n_parties)
    global_models = init_nets(args,1)
    global_model = global_models[0]
    global_para = global_model.state_dict()

    for net_id, net in nets.items():
        net.load_state_dict(global_para)

    n_comm_rounds = args.comm_round

    if args.load_model_file :
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    weights = []
    for i in range(args.party_per_round):
        weights.append([])
        weights[i] = 0
    print("start training")
    logger.info("start training")


    for round in range(args.comm_round):
        logger.info("in comm round:" + str(round))

        arr = np.arange(args.n_parties)
        np.random.shuffle(arr)
        selected = arr[:args.party_per_round]

        local_data_points = [len(train_dl_local_set[r]) for r in selected]

        index = 0
        for i in range(len(local_data_points)):
            weights[i] += local_data_points[i]

        global_para = global_model.state_dict()

        local_train_net(nets, selected, args, train_dl_set=train_dl_local_set, test_dl=test_dl)
        print("updating global")

        for idx in range(int(args.party_per_round)):
            net_id=selected[idx]
            net_para = nets[net_id].cpu().state_dict()
            weight = weights[idx] / sum(weights)
            if idx == 0:
                for key in net_para:
                    global_para[key] = net_para[key] * weight
            else:
                for key in net_para:
                    global_para[key] += net_para[key] * weight

        global_model.load_state_dict(global_para)


        logger.info('global n_test: %d' % len(test_dl))
        global_model.cuda()
        test_acc, _, = compute_accuracy(global_model, test_dl)
        logger.info('>> Global Model Test accuracy: %f' % test_acc)
        mkdirs(args.modeldir + 'fedec/')
        global_model.to('cpu')

        torch.save(global_model.state_dict(), args.modeldir + 'fedec/' + 'globalmodel'+str(args.n_label) + args.log_file_name + '.pth')

        for net_id, net in nets.items():
            net.load_state_dict(global_para)
        for idx in range(int(args.party_per_round)):
            weights[idx] = 0
