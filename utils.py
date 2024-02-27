import torch
import logging
import random
from models import *
from partition_data import *
import torchvision.models as models
def init_nets_proto(args,n_parties):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in ["cifar10"]:
        n_classes = 10
    elif args.dataset in ["cifar100"]:
        n_classes = 100
    elif args.dataset in ["femnist_by_writer"]:
        n_classes = 62

    for net_i in range(n_parties):
        if args.dataset in ("cifar10",  "cifar100"):
            net = model_cifar_proto(args, n_classes ,256)
        elif args.dataset == "femnist_by_writer":
            net = SimpleFemnist()
        nets[net_i] = net
    return nets


def init_nets(args,n_parties):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in ["cifar10","mnist","fmnist","SVHN"]:
        n_classes = 10
    elif args.dataset in ["cifar100"]:
        n_classes = 100
    elif args.dataset in ["femnist_by_writer"]:
        n_classes = 62

    for net_i in range(n_parties):
        if args.dataset in ("cifar10",  "cifar100"):

            if args.dataset=="cifar100":
                net=Wide_ResNet(num_classes=n_classes)
            else:
                net = model_cifar(args, n_classes)

        elif args.dataset in ["mnist","fmnist"]:

            args.model = "resnet18"
            net = model_cifar(args, n_classes)
        elif args.dataset in ["SVHN"]:

            args.model = "resnet18"
            net=model_cifar(args, n_classes)

        nets[net_i] = net
    return nets


def init_dataloader(args, net_dataidx_map=None):
    print("starting init dataloader")
    train_dl_local_set = []
    train_ds_local_set = []
    if args.dataset in ["femnist_by_writer","synthetic","shakespeare"]:
        for i in range(args.n_parties):
            train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args, identity=i)
            train_dl_local_set.append(train_dl_local)
            train_ds_local_set.append(train_ds_local)
            print(len(train_dl_local_set))
    elif args.dataset in ["fmnist","mnist","cifar10","cifar10l","cifar100","tiny-imagenet","SVHN"]:
        for i in range(args.n_parties):
            if net_dataidx_map==None:
                dataidxs=None
            else:
                dataidxs = net_dataidx_map[i]
            train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args,dataidxs=dataidxs)
            train_dl_local_set.append(train_dl_local)
            train_ds_local_set.append(train_ds_local)
            print(len(train_dl_local_set))
    elif args.dataset in ["PACS"]:
        train_X,train_Y,test_X,test_Y=build_PACS()
        for i in range(args.n_parties):
            train_ds=PACS_s3(client_id=i,X=train_X,Y=train_Y)

            train_dl_local = DataLoader(dataset=train_ds, batch_size=args.train_batchsize, drop_last=True, shuffle=True,
                                  prefetch_factor=4, persistent_workers=True, num_workers=4)
            train_dl_local_set.append(train_dl_local)
            train_ds_local_set.append(train_ds)
        test_ds_local=PACS_s3(client_id=-1,X=test_X,Y=test_Y)
        test_dl_local = DataLoader(dataset=test_ds_local, batch_size=args.train_batchsize, drop_last=True, shuffle=True,
                                    prefetch_factor=4, persistent_workers=True, num_workers=4)
    elif args.dataset in ["officehome"]:
        train_X, train_Y, test_X, test_Y = build_Officehome()
        for i in range(args.n_parties):
            train_ds = Officehome_s3(client_id=i, X=train_X, Y=train_Y)

            train_dl_local = DataLoader(dataset=train_ds, batch_size=args.train_batchsize, drop_last=True, shuffle=True,
                                        prefetch_factor=4, persistent_workers=True, num_workers=4)
            train_dl_local_set.append(train_dl_local)
            train_ds_local_set.append(train_ds)
        test_ds_local = Officehome_s3(client_id=-1, X=test_X, Y=test_Y)
        test_dl_local = DataLoader(dataset=test_ds_local, batch_size=args.train_batchsize, drop_last=True, shuffle=True,
                                   prefetch_factor=4, persistent_workers=True, num_workers=4)

    elif args.dataset in ["ISIC"]:
        for i in range(args.n_parties):
            train_ds = ISIC2019(identity=i)

            train_dl_local = DataLoader(dataset=train_ds, batch_size=args.train_batchsize, drop_last=True, shuffle=True,
                                        prefetch_factor=4, persistent_workers=True, num_workers=4)
            train_dl_local_set.append(train_dl_local)
            train_ds_local_set.append(train_ds)
        test_ds_local = ISIC2019(identity=-1, train=False)
        test_dl_local = DataLoader(dataset=test_ds_local, batch_size=args.train_batchsize, drop_last=True, shuffle=True,
                                   prefetch_factor=4, persistent_workers=True, num_workers=4)
    print("finishing init dataloader")
    return train_dl_local_set, test_dl_local,train_ds_local_set,test_ds_local

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def compute_accuracy(model, dataloader):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0

    criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):

            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            _,_,out = model(x)
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            loss_collector.append(loss.item())
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()


            pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
            true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)

    if was_training:
        model.train()

    return correct / float(total), avg_loss
def compute_accuracy_proto(model, dataloader):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0

    loss_collector = []

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):

            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            _,_,out = model(x)
            out = torch.log(out)
            loss = F.nll_loss(out, target)
            _, pred_label = torch.max(out.data, 1)
            loss_collector.append(loss.item())
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()


            pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
            true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)

    if was_training:
        model.train()

    return correct / float(total), avg_loss