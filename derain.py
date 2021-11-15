from __future__ import print_function

import argparse

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
from torch import optim

cudnn.benchmark = True
cudnn.fastest = True
from torch.autograd import Variable
import torch.nn.functional as func

from utils.SSIM import SSIM
from utils.misc import *
import models.derain_dense_relu as net

parser = argparse.ArgumentParser()
parser.add_argument('--epochTrainingNum', required=False, default=95, help='Number of training epochs')
parser.add_argument('--dataset', required=False, default='pix2pix', help='')
parser.add_argument('--dataroot', required=False, default='./data/DID-MDN-training/train2')
parser.add_argument('--valDataroot', required=False, default='./data/DID-MDN-training/train2',
                    help='path to val dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=120, help='input batch size')
parser.add_argument('--originalSize', type=int,
                    default=512, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
                    default=128, help='the height / width of the cropped input image to network')
parser.add_argument('--inputChannelSize', type=int,
                    default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
                    default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
parser.add_argument('--annealEvery', type=int, default=400, help='epoch to reaching at learning rate of 0')
parser.add_argument('--lambdaGAN', type=float, default=0.01, help='lambdaGAN')
parser.add_argument('--lambdaIMG', type=float, default=1, help='lambdaIMG')
parser.add_argument('--poolSize', type=int, default=50,
                    help='Buffer size for storing previously generated samples from G')
parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in D')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=500,
                    help='interval for evauating(generating) images from valDataroot')
opt = parser.parse_args(args=[])


def calculate_loss(a, input_cpu, kk3, kk4, re, residual_cpu, residualimg, rm, rx3, rx4, ry3, ry4, target):
    a3 = func.avg_pool2d(a, 2)
    a3 = a3.cuda()
    a3 = Variable(a3)
    d3 = torch.div(0.05, torch.abs(a3 - rm))
    d3 = d3.cuda()
    d4 = d3.detach()
    L_img4 = criterionCAE(re, d4)
    L_img33 = criterionCAE(a3 * rm, a3 * a3)
    L_img3 = criterionCAE(rm, a3)
    L_img6 = criterionCAE(rx3, ry3)
    L_img7 = criterionCAE(rx4, ry4)
    cleanimg1 = input_cpu - residualimg
    L_img9 = criterion(cleanimg1, target)
    L_img10 = criterionCAE(residualimg, residual_cpu)
    L_img = 0.6 * (L_img3 + L_img6 + L_img7 + 0 * L_img4 + 0.01 * torch.mean(kk3 * kk3) + 0.01 * torch.mean(
        kk4 * kk4)) - L_img9 + L_img10
    losses = [L_img4, L_img33, L_img3, L_img6, L_img7, L_img9, L_img10]
    return L_img, L_img10


def train(model, optimizerG, epochTN, outputChannelSize, input1, target, k=0):
    losses = []
    for epoch in range(0, epochTN):

        if ((epoch + 1) % 15) == 0:
            k = 1
        if ((epoch + 1) % 30) == 0:
            k = 0

        if ((epoch + 1) % 30) == 0:
            adjust_learning_rate(optimizerG, opt.lrG, epoch, None, 100)

        dataloader = getLoader(opt.dataset,
                               opt.dataroot,
                               opt.originalSize,
                               opt.imageSize,
                               opt.batchSize,
                               opt.workers,
                               mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                               split='train',
                               shuffle=True,
                               )
        print(f"Total number of images: {len(dataloader)}")
        for i, data in enumerate(dataloader, 0):
            input_cpu, target_cpu, label_cpu = data
            batch_size = target_cpu.size(0)

            residual_cpu = input_cpu - target_cpu

            target_cpu, input_cpu, residual_cpu = target_cpu.float().cuda(), input_cpu.float().cuda(), residual_cpu.float().cuda()
            residual_cpu = Variable(residual_cpu)

            target.resize_as_(target_cpu).copy_(target_cpu)
            input1.resize_as_(input_cpu).copy_(input_cpu)

            model.zero_grad()

            residualimg, rx3, ry3, rx4, ry4, rm, re, kk3, kk4 = model(input1, residual_cpu)

            a = torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)

            a = Variable(a)
            a = a.cuda()
            a.resize_as_(residual_cpu).copy_(residual_cpu)

            L_img, L_img10 = calculate_loss(a, input_cpu, kk3, kk4, re, residual_cpu, residualimg, rm, rx3, rx4, ry3,
                                            ry4, target)

            L_img.backward()

            optimizerG.step()

            loss = L_img10.item()
            print(f"Epoch: {epoch} Image: {i} L1Loss:{loss}")

        if epoch % 1 == 0:
            print('Finish training epoch %d' % epoch)
            torch.save(model.state_dict(), '%s/netG1_epoch_%d.pth' % (opt.exp, epoch))


if __name__ == '__main__':
    trainLogger = open('%s/train.log' % opt.exp, 'w')
    create_exp_dir(opt.exp)
    # get dataloader
    dataloader = getLoader(opt.dataset,
                           opt.dataroot,
                           opt.originalSize,
                           opt.imageSize,
                           opt.batchSize,
                           opt.workers,
                           mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                           split='train',
                           shuffle=True,
                           )

    model = net.Dense_rainall()
    # model.load_state_dict(torch.load('./sample1/netG1_epoch_95.pth'))
    print("Train logger")
    model.train()
    model.cuda()
    model = torch.nn.DataParallel(model, [0])
    # # model.load_state_dict(torch.load('./sample1/netG1_epoch_95.pth'))
    print("Criterions")
    criterion_class = nn.CrossEntropyLoss()
    criterionCAE = nn.L1Loss()
    criterionCAB = nn.MSELoss()
    criterion = SSIM()
    criterion.cuda()
    criterionCAE.cuda()
    criterion_class.cuda()

    print("Tensor creation")
    label_d = torch.FloatTensor(opt.batchSize)
    target = torch.FloatTensor(opt.batchSize, opt.outputChannelSize, opt.imageSize, opt.imageSize)
    input1 = torch.FloatTensor(opt.batchSize, opt.inputChannelSize, opt.imageSize, opt.imageSize)
    target, input1 = target.cuda(), input1.cuda()
    label_d = Variable(label_d.cuda())
    target = Variable(target)
    input1 = Variable(input1)

    optimizerG = optim.Adam(model.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999), weight_decay=0.0001)
    train(model, optimizerG, int(opt.epochTrainingNum), opt.outputChannelSize, input1, target)

    trainLogger.close()
