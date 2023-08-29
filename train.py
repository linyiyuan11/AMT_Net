# coding=utf-8
import os
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import DA_DatasetFromFolder, TestDatasetFromFolder, LoadDatasetFromFolder
import numpy as np
import random
from model.network import CDNet
from train_options import parser
import itertools
from loss.losses import cross_entropy
import time
from collections import OrderedDict
import ever as er

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed_torch(2022)

COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
)

def val(mloss):
    CDNet.eval()
    with torch.no_grad():
        val_bar = tqdm(test_loader)
        valing_results = {'batch_sizes': 0, 'IoU': 0}

        metric_op = er.metric.PixelMetric(2, logdir=None, logger=None)
        for hr_img1, hr_img2, label, name in val_bar:

            valing_results['batch_sizes'] += args.val_batchsize
            hr_img1 = hr_img1.to(device, dtype=torch.float)
            hr_img2 = hr_img2.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float)

            label = torch.argmax(label, 1).unsqueeze(1).float()

            cd_map, _, _ = CDNet(hr_img1, hr_img2)

            cd_map = torch.argmax(cd_map, 1).unsqueeze(1).float()

            gt_value = (label > 0).float()
            prob = (cd_map > 0).float()
            prob = prob.cpu().detach().numpy()

            gt_value = gt_value.cpu().detach().numpy()
            gt_value = np.squeeze(gt_value)
            result = np.squeeze(prob)
            metric_op.forward(gt_value, result)
        re = metric_op.summary_all()

    test_loss = re.rows[1][1]
    if test_loss > mloss or epoch == 1:
        torch.save(CDNet.state_dict(), args.model_dir +'_'+ str(test_loss) +'_best.pth')
    CDNet.train()
    return test_loss

def train_epoch():
    CDNet.train()
    for hr_img1, hr_img2, label in train_bar:
        running_results['batch_sizes'] += args.batchsize

        hr_img1 = hr_img1.to(device, dtype=torch.float)
        hr_img2 = hr_img2.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.float)
        label = torch.argmax(label, 1).unsqueeze(1).float()

        result1, result2, result3 = CDNet(hr_img1, hr_img2)

        CD_loss = CDcriterionCD(result1, label) + CDcriterionCD(result2, label) + CDcriterionCD(result3, label)

        CDNet.zero_grad()
        CD_loss.backward()
        optimizer.step()

        running_results['CD_loss'] += CD_loss.item() * args.batchsize

        train_bar.set_description(
            desc='[%d/%d] loss: %.4f' % (
                epoch, args.num_epochs,
                running_results['CD_loss'] / running_results['batch_sizes'],))


if __name__ == '__main__':
    mloss = 0

    # load data
    train_set = DA_DatasetFromFolder(args.hr1_train, args.hr2_train, args.lab_train, crop=False)
    test_set = TestDatasetFromFolder(args, args.hr1_test, args.hr2_test, args.lab_test)

    train_loader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batchsize, shuffle=True)
    test_loader = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=args.val_batchsize, shuffle=True)

    # define model
    CDNet = CDNet(img_size = args.img_size).to(device, dtype=torch.float)

    # set optimization
    optimizer = optim.AdamW(itertools.chain(CDNet.parameters()), lr= args.lr, betas=(0.9, 0.999),weight_decay=0.01)
    CDcriterionCD = cross_entropy().to(device, dtype=torch.float)

    # training
    for epoch in range(1, args.num_epochs + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'SR_loss':0, 'CD_loss':0, 'loss': 0 }
        train_epoch()
        mloss = val(mloss)


