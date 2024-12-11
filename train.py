import crl_utils
import utils

import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
# import pandas as pd

# def kl_loss(logits, calibrated_logits):
#     """KL Divergence Loss Calculation"""
#     return F.kl_div(
#         F.log_softmax(calibrated_logits, dim=-1), 
#         F.softmax(logits, dim=-1), 
#         reduction='batchmean'
#     )

def rank_reg(output, target):
    ranks = []
    target_copy = target.cpu()

    tgt_idx = [(i, target_copy[i]) for i in range(len(target_copy))]# gets the indices of the target classification in the form of [[i,j],[i,l]] where j and l are the positions of the targets
    # Use the above score to pinpoint True positives in the batch score matrix
    # print(f"Target Index:\n{tgt_idx}\n\nTarget Copy:\n{target_copy}")
    
    output_copy = output.cpu().data.numpy().copy()
    # df = pd.DataFrame(output)

    # Calculate ranked outputs by class and isolate TP ranks
    ranked_output = np.zeros_like(output_copy)
    transposed_output = output_copy.T
    temp = (-transposed_output).argsort()
    ranked_output = temp.argsort().T
    # print(f"Ranked Output:\n{ranked_output}")
    # ranked_output = df.rank(0, ascending=False).astype(int).values
    for tgt in tgt_idx:
        # print(ranked_output[target_copy])
        ranks.append(ranked_output[tgt]**2)

    # print(f"\n\nRanks:\n{ranks}")
    loss = sum(ranks)/len(ranks)
    # print(loss)

    return loss

def train(loader, model, criterion_cls, criterion_ranking, optimizer, epoch, history, logger, rank, args, val_loader):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    total_losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    cls_losses = utils.AverageMeter()
    ranking_losses = utils.AverageMeter()
    end = time.time()
    # kl_losses = utils.AverageMeter()

    model.train()
    for i, (input, target, idx) in enumerate(loader):
        data_time.update(time.time() - end)
        input, target = input.cuda(), target.cuda()
        # compute output
        
        output = model(input)
        # logits, output = model(input)

        # compute ranking target value normalize (0 ~ 1) range
        # max(softmax)
        if args.rank_target == 'softmax':
            conf = F.softmax(output, dim=1)
            confidence, _ = conf.max(dim=1)
        # entropy
        elif args.rank_target == 'entropy':
            if args.data == 'cifar100':
                value_for_normalizing = 4.605170
            else:
                value_for_normalizing = 2.302585
            confidence = crl_utils.negative_entropy(output,
                                                    normalize=True,
                                                    max_value=value_for_normalizing)
        # margin
        elif args.rank_target == 'margin':
            conf, _ = torch.topk(F.softmax(output), 2, dim=1)
            conf[:,0] = conf[:,0] - conf[:,1]
            confidence = conf[:,0]

        # make input pair
        rank_input1 = confidence
        rank_input2 = torch.roll(confidence, -1)
        idx2 = torch.roll(idx, -1)

        # calc target, margin
        rank_target, rank_margin = history.get_target_margin(idx, idx2)
        rank_target_nonzero = rank_target.clone()
        rank_target_nonzero[rank_target_nonzero == 0] = 1
        rank_input2 = rank_input2 + rank_margin / rank_target_nonzero

        # ranking loss
        if rank=="rank_loss":
            ranking_loss = criterion_ranking(rank_input1,
                                            rank_input2,
                                            rank_target)
        
        elif rank=="rank_reg":
            ranking_loss = rank_reg(output, target)

        #KL-loss
        # kld_loss = kl_loss(logits, output)

        # total loss
        cls_loss = criterion_cls(output, target)
        ranking_loss = args.rank_weight * ranking_loss
        loss = cls_loss + ranking_loss
        # loss = cls_loss + ranking_loss + 0.1 * kld_loss

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # record loss and accuracy
        prec, correct = utils.accuracy(output, target)
        total_losses.update(loss.item(), input.size(0))
        cls_losses.update(cls_loss.item(), input.size(0))
        ranking_losses.update(ranking_loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))
        # kl_losses.update(kld_loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Cls Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                  'Rank Loss {rank_loss.val:.4f} ({rank_loss.avg:.4f})\t'
                  'Prec {top1.val:.2f}% ({top1.avg:.2f}%)'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=total_losses, cls_loss=cls_losses,
                   rank_loss=ranking_losses,top1=top1))

        # correctness count update
        history.correctness_update(idx, correct, output)
    # max correctness update
    history.max_correctness_update(epoch)

    model.eval()
    with torch.no_grad():
        total_loss = 0
        for i, (input, target, idx) in enumerate(tqdm(val_loader)):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            
            if args.rank_target == 'softmax':
                conf = F.softmax(output, dim=1)
                confidence, _ = conf.max(dim=1)
            # entropy
            elif args.rank_target == 'entropy':
                if args.data == 'cifar100':
                    value_for_normalizing = 4.605170
                else:
                    value_for_normalizing = 2.302585
                confidence = crl_utils.negative_entropy(output,
                                                        normalize=True,
                                                        max_value=value_for_normalizing)
            # margin
            elif args.rank_target == 'margin':
                conf, _ = torch.topk(F.softmax(output), 2, dim=1)
                conf[:,0] = conf[:,0] - conf[:,1]
                confidence = conf[:,0]

            # make input pair
            rank_input1 = confidence
            rank_input2 = torch.roll(confidence, -1)
            idx2 = torch.roll(idx, -1)

            # calc target, margin
            rank_target, rank_margin = history.get_target_margin(idx, idx2)
            rank_target_nonzero = rank_target.clone()
            rank_target_nonzero[rank_target_nonzero == 0] = 1
            rank_input2 = rank_input2 + rank_margin / rank_target_nonzero

            # ranking loss
            if rank=="rank_loss":
                ranking_loss = criterion_ranking(rank_input1,
                                                rank_input2,
                                                rank_target)
            
            elif rank=="rank_reg":
                ranking_loss = rank_reg(output, target)

            #KL-loss
            # kld_loss = kl_loss(logits, output)

            # total loss
            cls_loss = criterion_cls(output, target)
            ranking_loss = args.rank_weight * ranking_loss
            loss = cls_loss + ranking_loss
            total_loss += loss
        
        print(f"\nAverage Validation Loss: {total_loss/len(val_loader)}\n")
        logger.write([epoch, total_losses.avg, cls_losses.avg, ranking_losses.avg, top1.avg, float(total_loss/len(val_loader))])