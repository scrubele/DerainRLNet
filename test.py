from __future__ import print_function

import argparse
import time

from PIL import Image

import models.derain_dense_relu_test as net
from utils.misc import *

# Pre-defined Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='pix2pix', help='')
parser.add_argument('--dataroot', required=False, default='/content/gdrive/MyDrive/1/')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--originalSize', type=int,
                    default=512, help='the height / width of the original input image')
parser.add_argument('--image_size', type=int,
                    default=512, help='the height / width of the cropped input image to network')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--exp', default='sample6', help='folder to output images and model checkpoints')
parser.add_argument('--output_directory', type=str, default='./result/', help='output images directory')
parser.add_argument('--model_path', type=str,
                    default='/content/gdrive/MyDrive/DerainRLNet/derain/sample/netG1_epoch_12.pth',
                    help='path to the .pth trained model')
IS_COLAB = True


def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.div_(2).add_(0.5)

    return img


def norm_range(t, range):
    img = norm_ip(t, t.min(), t.max())
    return img


def save_result(i, im2, output_directory):
    directory1 = '%s' % output_directory
    if not os.path.exists(directory1):
        os.makedirs(directory1)
    filename1 = output_directory + str(i) + '.png'
    im2.save(filename1)


def test_model(model, validation_data_loader, image_size, output_directory):
    # Begin Testing
    for epoch in range(1):
        start_time = time.time()
        for i, data in enumerate(validation_data_loader, 0):
            with torch.no_grad():
                print('Image:' + str(i))

                input = torch.FloatTensor(1, 3, image_size, image_size).cuda()
                input_cpu, target_cpu, label_cpu, w11, h11 = data
                residual_cpu = input_cpu - target_cpu
                input_cpu, residual_cpu = input_cpu.float().cuda(), residual_cpu.float().cuda()

                torch.cuda.synchronize()
                start_i_time = time.time()
                try:
                    residual_img = model(input_cpu)
                    torch.cuda.synchronize()
                    finish_i_time = time.time()
                    print("Time:", finish_i_time - start_i_time)

                    a = input_cpu - residual_img

                    tensor = a.data.cpu()
                    tensor = torch.squeeze(tensor)

                    img_array = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                    img = Image.fromarray(img_array)
                    img = img.resize((w11, h11), Image.NEAREST)

                    save_result(i, img, output_directory)
                except:
                    print(f"Error: {i}")
                    pass
        finish_time = time.time()
        print("Total time:", (finish_time - start_time) / 100)


if __name__ == '__main__':
    if IS_COLAB:
        opt = parser.parse_args(args=[])

    create_exp_dir(opt.exp)

    validation_data_loader = getLoader(opt.dataset,
                                       opt.dataroot,
                                       opt.image_size,  # opt.originalSize,
                                       opt.image_size,
                                       opt.batch_size,
                                       opt.workers,
                                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                                       split='validation',
                                       shuffle=False
                                       )

    model = net.Dense_rainall().cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(opt.model_path))
    model.eval()

    test_model(model, validation_data_loader, image_size=opt.image_size, output_directory=opt.output_directory)
