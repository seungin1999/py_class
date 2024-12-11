from model import resnet
from model import densenet_BC
from model import vgg

import data_with_validation as dataset
import crl_utils
import metrics
import utils
import train

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from model import hts
from plot_roc_auc import roc_auc_value
import torch.nn.functional as F
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description='Confidence Aware Learning')
parser.add_argument('--epochs', default=300, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
parser.add_argument('--data', default='cifar10', type=str, help='Dataset name to use [cifar10, cifar100, svhn]')
parser.add_argument('--model', default='res', type=str, help='Models name to use [res, dense, vgg]')
parser.add_argument('--rank_target', default='softmax', type=str, help='Rank_target name to use [softmax, margin, entropy]')
parser.add_argument('--rank_weight', default=1.0, type=float, help='Rank loss weight')
parser.add_argument('--data_path', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--save_path', default='./test/', type=str, help='Savefiles directory')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--rank_type', '-r', default="rank_loss", type=str, help='Selects Rank Loss Type (default: rank_loss, alt: rank_reg)')
parser.add_argument('--temp', default=1.0, type=float)

args = parser.parse_args()

def main():
    # set GPU ID
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True

    set_seed(42)

    # check save path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # make dataloader
    train_loader, test_loader, \
    test_onehot, test_label, val_loader = dataset.get_loader(args.data,
                                                 args.data_path,
                                                 args.batch_size,
                                                 imbalanced_cifar=True)

    # set num_class
    if args.data == 'cifar100':
        num_class = 100
    else:
        num_class = 10

    # set num_classes
    model_dict = {
        "num_classes": num_class,
    }

    # set model
    # if args.model == 'res':
    #     model = resnet.resnet110(**model_dict).cuda()
    # elif args.model == 'dense':
    #     model = densenet_BC.DenseNet3(depth=100,
    #                                   num_classes=num_class,
    #                                   growth_rate=12,
    #                                   reduction=0.5,
    #                                   bottleneck=True,
    #                                   dropRate=0.0).cuda()
    # elif args.model == 'vgg':
    #     model = vgg.vgg16(**model_dict).cuda()

    # model = TemperatureScaling(model).cuda()

    if args.model=='res':
        model = hts.ResNetWithHTS(num_class, **model_dict).cuda()

    # set criterion
    cls_criterion = nn.CrossEntropyLoss().cuda()
    ranking_criterion = nn.MarginRankingLoss(margin=0.0).cuda()

    # set optimizer (default:sgd)
    optimizer = optim.SGD(model.parameters(),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=0.0001,
                          nesterov=False)

    # set scheduler
    scheduler = MultiStepLR(optimizer,
                            milestones=[150,250],
                            gamma=0.1)

    # make logger
    train_logger = utils.Logger(os.path.join(save_path, 'train.log'))
    result_logger = utils.Logger(os.path.join(save_path, 'result.log'))

    # make History Class
    correctness_history = crl_utils.History(len(train_loader.dataset))

    # start Train
    for epoch in range(1, args.epochs + 1):
        
        train.train(train_loader,
                    model,
                    cls_criterion,
                    ranking_criterion,
                    optimizer, 
                    epoch,
                    correctness_history,
                    train_logger,
                    args.rank_type,
                    args,
                    val_loader
                    )
        scheduler.step()
        
        # save model
        if epoch == args.epochs:
            torch.save(model.state_dict(),
                       os.path.join(save_path, f'latest_model.pth'))
    # finish train
    
    # calc measure
    
    acc, aurc, eaurc, aupr, fpr, ece, nll, brier = metrics.calc_metrics(test_loader,
                                                                        test_label,
                                                                        test_onehot,
                                                                        model,
                                                                        cls_criterion)
    
    # result write
    result_logger.write([acc, aurc*1000, eaurc*1000, aupr*100, fpr*100, ece*100, nll*10, brier*100])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    probabilities = []
    list_correct = []
    labels= []
    with torch.no_grad():
        for inputs, label, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            probabilities.extend(probs.cpu().numpy())
            labels.extend(label.cpu().numpy())
            pred = outputs.data.max(1, keepdim=True)[1]
            label = label.cuda()
                
            for j in range(len(pred)):
                if pred[j] == label[j]:
                    cor = 1
                else:
                    cor = 0
                list_correct.append(cor)

    probabilities = np.array(probabilities)

    AUROC = roc_auc_value(list_correct, probabilities)
    
    print("AUROC:", AUROC)

if __name__ == "__main__":
    main()



