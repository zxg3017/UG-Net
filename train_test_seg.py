'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import os
import csv
import cv2

import argparse
from tqdm import tqdm
from my_dataset.my_skin_lesion_datasets import MyDataSet_seg, MyValDataSet_seg, MyTestDataSet_seg
import numpy as np
from sklearn.metrics import accuracy_score
from models.metrics import Miou
import torch.nn.functional as F
import warnings

import utils as utils
import models.loss.Loss_all as Loss
from models.my_H_Net_model.H_Net_Efficient import Coarse_SN_Efficient_b3_DAC_CAC_input_cat
from torch.utils import data
from data.create_dataset_with_coarse_mask import CreateDataset
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Segmentation method training')
parser.add_argument('--resize', default=256, type=int, help='resize shape')
parser.add_argument('--batch_size', default=16,type=int,help='batchsize')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--start_epoch', '-s', default=0, type=int, help='start epoch')
parser.add_argument('--end_epoch', '-e', default=200, type=int, help='end epoch')
parser.add_argument('--times', '-t', default=1, type=int, help='val')
parser.add_argument('--device', default='cuda', type=str, help='use cuda')
parser.add_argument('--tb_path', type=str, default='log/Unet', help='tensorboard path')
parser.add_argument('--checkpoint', type=str, default='checkpoint/Nn-Net/wbc/', help='checkpoint path')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
torch.backends.cudnn.enabled =True
tb_path = args.tb_path
if not os.path.exists(tb_path):
    os.mkdir(tb_path)
device = args.device # 是否使用cuda
# model_urls = {'deeplabv3plus_xception': 'models_00/pretrained_models/deeplabv3plus_xception_VOC2012_epoch46_all.pth'}
result_path='/mnt/ai2019/zxg_FZU/dataset/Nn-Net_result/wbc_seg/'
result_roc_path='/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/CCE-NET/test/Nn-Net/'


os.environ["CUDA_VISIBLE_DEVICES"] = '2'
warnings.filterwarnings("ignore")

best_miou = 0  # best test accuracy
EPS = 1e-12
start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
times = args.times  # 验证次数
checkpoint_path = args.checkpoint + 'wbc_Nn_Net_seg_with_DAC_CAC_loss-input-cat_zxg.pth'
# Data                        csv_cns_cat_ckpt_v2.pth
print('==> Preparing data..')

############# Load training and validation data
path_train_img = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/train/images/'
path_train_mask = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/train/masks/'
path_train_coarsemask = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/CCE-NET/train/coarse_mask3/'
trainset = CreateDataset(img_paths=path_train_img, label_paths=path_train_mask, paths_coarseMask=path_train_coarsemask,
                         resize=args.resize, phase='train', aug=True)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

path_val_img = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/val/images/'
path_val_mask = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/val/masks/'
path_val_coarsemask = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/CCE-NET/val/coarse_mask3/'
valset = CreateDataset(img_paths=path_val_img, label_paths=path_val_mask, paths_coarseMask=path_val_coarsemask,
                         resize=args.resize, phase='val', aug=False)
valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

############# Load testing data
path_test_img = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/test/images/'
path_test_mask = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/test/masks/'
path_test_coarsemask = '/mnt/ai2019/zxg_FZU/dataset/zxg_wbc_seg/seg/test/masks/'
testset = CreateDataset(img_paths=path_test_img, label_paths=path_test_mask, paths_coarseMask=path_test_coarsemask,
                         resize=args.resize, phase='val', aug=False)
testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)



# Model
print('==> Building model..')

net = Coarse_SN_Efficient_b3_DAC_CAC_input_cat(4,2)
# net = deeplabv3plus(num_classes=2)
print("param size = %fMB", utils.count_parameters_in_MB(net))

net = net.to(device)
# print(args.resume)
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.checkpoint), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    best_miou = checkpoint['miou']
    start_epoch = checkpoint['epoch']

# criterion = nn.NLLLoss2d().to(device)
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200,eta_min=1e-8)

ce_Loss = Loss.CrossEntropyLoss2D().to(device)
dice_Loss = Loss.myDiceLoss(2).to(device)
softmax_2d = nn.Softmax2d()

net.train()
net.float()
def train_val():
    with SummaryWriter(tb_path) as write:
        train_step = 0
        for epoch in range(start_epoch, args.end_epoch):
            with tqdm(total=len(trainloader), ncols=120, ascii=True) as t:
                net.train()
                train_loss = 0
                train_miou = 0

                for batch_idx, (inputs, coarsemask, label0, label, name) in enumerate(trainloader):
                    t.set_description("Train(Epoch{}/{})".format(epoch, args.end_epoch))
                    inputs, label, coarsemask = inputs.to(device), label.to(device), coarsemask.to(device)
                    coarsemask = coarsemask.unsqueeze(1).cuda()
                    label = label.long()
                    inputs = torch.cat([inputs, coarsemask], dim=1)

                    out = net(inputs)
                    out = torch.log(softmax_2d(out) + EPS)
                    ce_loss = ce_Loss(torch.log(softmax_2d(out) + EPS), label)
                    dice_loss = dice_Loss(torch.log(softmax_2d(out) + EPS), label)
                    loss = 0.4 * ce_loss + 0.6 * dice_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    predicted = out.argmax(1)
                    train_miou += Miou.calculate_miou(predicted, label, 2).item()

                    write.add_scalar('Train_loss', train_loss / (batch_idx + 1), global_step=train_step)
                    write.add_scalar('Train_Miou', 100. * (train_miou / (batch_idx + 1)), global_step=train_step)
                    train_step += 1
                    t.set_postfix(loss='{:.3f}'.format(train_loss / (batch_idx + 1)),
                                  train_miou='{:.2f}%'.format(100. * (train_miou / (batch_idx + 1))))
                    t.update(1)

                scheduler.step()
            if epoch % times == 0:
                global best_miou
                net.eval()
                val_loss = 0
                val_miou = 0
                with torch.no_grad():
                    with tqdm(total=len(valloader), ncols=120, ascii=True) as t:
                        for batch_idx, (inputs, coarsemask, label0, label, name) in enumerate(valloader):
                            t.set_description("Val(Epoch {}/{})".format(epoch, args.end_epoch))
                            inputs, label, coarsemask = inputs.cuda(), label.cuda(), coarsemask.cuda()
                            coarsemask = coarsemask.unsqueeze(1).cuda()
                            inputs = torch.cat([inputs, coarsemask], dim=1)

                            out = net(inputs)
                            out = torch.log(softmax_2d(out) + EPS)
                            label = label.long()
                            predicted = out.argmax(1)
                            val_miou += Miou.calculate_miou(predicted, label, 2).item()

                            t.set_postfix(val_miou='{:.2f}%'.format(100. * (val_miou / (batch_idx + 1))))
                            t.update(1)
                        write.add_scalar('Val_loss', val_loss / (batch_idx + 1), global_step=train_step)
                        write.add_scalar('Val_miou', 100. * (val_miou / (batch_idx + 1)), global_step=train_step)
                        # Save checkpoint.
                    val_miou = 100. * (val_miou / (batch_idx + 1))
                    if val_miou > best_miou:
                        print('Saving..')
                        state = {
                            'net': net.state_dict(),
                            'miou': val_miou,
                            'epoch': epoch,
                        }
                        if not os.path.isdir(args.checkpoint):
                            os.mkdir(args.checkpoint)
                        torch.save(state, checkpoint_path)
                        best_miou = val_miou

def test_data():
    net.eval()

    net.load_state_dict(torch.load(
        '/mnt/ai2019/zxg_FZU/seg_and_cls_projects/my_secode_paper_source_code/checkpoint/Nn-Net/wbc/wbc_Nn_Net_seg_with_DAC_CAC_loss-input-cat_zxg.pth')['net'],
                        strict=True)

    with torch.no_grad():
        miou = []
        mdice = []
        precision = []
        recall = []
        F1score = []
        PA = []
        dice = []
        sen = []
        spe = []
        acc = []
        jac_score = []
        metrics_for_csv = []

        for batch_idx, (image, coarsemask, label0, label, img_path) in enumerate(testloader):
            batch_idx += 1
            image, coarsemask, label = image.to(device), coarsemask.to(device), label.to(device)
            targets = label.long()
            coarsemask = coarsemask.unsqueeze(1).cuda()
            ##
            rot_90 = torch.rot90(image, 1, [2, 3])
            rot_180 = torch.rot90(image, 2, [2, 3])
            rot_270 = torch.rot90(image, 3, [2, 3])
            hor_flip = torch.flip(image, [-1])
            ver_flip = torch.flip(image, [-2])
            image = torch.cat([image, rot_90, rot_180, rot_270, hor_flip, ver_flip], dim=0)

            rot_90_cm = torch.rot90(coarsemask, 1, [2, 3])
            rot_180_cm = torch.rot90(coarsemask, 2, [2, 3])
            rot_270_cm = torch.rot90(coarsemask, 3, [2, 3])
            hor_flip_cm = torch.flip(coarsemask, [-1])
            ver_flip_cm = torch.flip(coarsemask, [-2])
            coarsemask = torch.cat([coarsemask, rot_90_cm, rot_180_cm, rot_270_cm, hor_flip_cm, ver_flip_cm], dim=0)
            image = torch.cat([image,coarsemask], dim=1)
            out= net(image)
            pred = torch.log(softmax_2d(out) + EPS)
            #
            pred = pred[0:1] + torch.rot90(pred[1:2], 3, [2, 3]) + torch.rot90(pred[2:3], 2, [2, 3]) + torch.rot90(
                pred[3:4], 1, [2, 3]) + torch.flip(pred[4:5], [-1]) + torch.flip(pred[5:6], [-2])
            predicted = pred.argmax(1)

            ##########################
            pred1 = torch.softmax(pred[0], dim=0).cpu().data.numpy()
            pred_arg = np.int16(np.argmax(pred1, axis=0))

            mask = label0[0].data.numpy()
            test_mask = np.int64(mask > 0)
            # y_pred
            y_true_f = test_mask.reshape(test_mask.shape[0] * test_mask.shape[1], order='F')
            y_pred_f = pred_arg.reshape(pred_arg.shape[0] * pred_arg.shape[1], order='F')

            intersection = np.float(np.sum(y_true_f * y_pred_f))
            dice.append((2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f)))
            sen.append(intersection / np.sum(y_true_f))
            intersection0 = np.float(np.sum((1 - y_true_f) * (1 - y_pred_f)))
            spe.append(intersection0 / np.sum(1 - y_true_f))
            acc.append(accuracy_score(y_true_f, y_pred_f))
            jac_score.append(intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection))
            #########################

            mdice.append(Miou.calculate_mdice(predicted, targets, 2))
            miou.append(Miou.calculate_miou(predicted, targets, 2))
            precision.append(Miou.pre(predicted, targets))
            recall.append(Miou.recall(predicted, targets))
            F1score.append(Miou.F1score(predicted, targets))
            PA.append(Miou.Pa(predicted, targets))

            predict = predicted.squeeze(0)
            img_np = predict.cpu().numpy()  # np.array
            size = img_np.shape[:2]

            img_np = (img_np * 255).astype('uint8')
            i = img_path[0].split('/')[-1]
            cv2.imwrite(os.path.join(result_roc_path, i), img_np)

            logi = F.softmax(torch.squeeze(pred), dim=0)[1].to('cpu').numpy()
            # cv2.imwrite(os.path.join(result_roc_path, i),cv2.resize((logi * 255),size[::-1], interpolation=cv2.INTER_NEAREST))

        mdice = format(np.nanmean(np.array(mdice)),'.4f')
        miou = format(np.nanmean(np.array(miou)),'.4f')
        precision = format(np.nanmean(np.array(precision)),'.4f')
        recall = format(np.nanmean(np.array(recall)),'.4f')
        F1score = format(np.nanmean(np.array(F1score)),'.4f')
        PA = format(np.array(torch.mean(torch.tensor(PA))),'.4f')
        dice = format(np.nanmean(np.array(dice)),'.4f')
        sen = format(np.nanmean(np.array(sen)),'.4f')
        spe = format(np.nanmean(np.array(spe)),'.4f')
        acc = format(np.nanmean(np.array(acc)),'.4f')
        jac_score = format(np.nanmean(np.array(jac_score)),'.4f')
        metrics_for_csv.append(['mdice', 'miou', 'precision', 'recall', 'F1score', 'PA', 'dice', 'sen', 'spe','acc','jac_score'])

        metrics_for_csv.append([mdice, miou, precision, recall, F1score, PA, dice, sen, spe, acc, jac_score])


        # results_file = open(result_path + 'wbc_seg_Nn_Net_with_DAC_CAC_loss-input-cat-zxg.csv', 'w', newline='')
        # csv_writer = csv.writer(results_file, dialect='excel')
        # for row in metrics_for_csv:
        #     csv_writer.writerow(row)

if __name__ == '__main__':
    # train_val()
    test_data()
