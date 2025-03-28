import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import cv2
from net.bgnet import Net
from Utils.data_val import test_dataset
import configparser

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


data_names = []
data_names.append('CAMO')
data_names.append('CHAMELEON')
data_names.append('COD10K')
data_names.append('NC4K')



parser = argparse.ArgumentParser()
parser.add_argument('--trainsize', type=int, default=448, help='testing size')
parser.add_argument('--pth_path', type=str, default='./checkpoints/Net_cod/Net_best.pth')
#parser.add_argument('--pth_path', type=str, default='./checkpoints/model_new/Net_best.pth')
parser.add_argument('--test_dataset_path', type=str, default='./data/TestDataset/')
    # parser.add_argument('--net_channel', type=int, default=config['Comm'].getint('net_channel'))
opt = parser.parse_args()


for _data_name in data_names:
    data_path = opt.test_dataset_path + '/{}/'.format(_data_name)
    save_path = './res/{}_3/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
    os.makedirs(save_path, exist_ok=True)

    model = Net()

        # state_dict = torch.load(pth_path, map_location=torch.device('cpu'))
        # model.load_state_dict(state_dict, strict=False)

    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(opt.pth_path).items()})
    model.cuda()
    model.eval()

    image_root = '{}/Image/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.trainsize)

    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        print('> {} - {}'.format(_data_name, name))

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        _, _, res, _ = model(image)

        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
            # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, res * 255)
