import argparse
file_train_path_time1 = '/./train/time1'
file_train_path_time2 = '/./train/time2'
file_train_path_label = '/./train/label'

file_test_path_time1 = '/./test/time1'
file_test_path_time2 = '/./test/time2'
file_test_path_label = '/./test/label'

#training options
parser = argparse.ArgumentParser(description='Training Change Detection Network')

# training parameters
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--val_batchsize', default=8, type=int, help='batchsize for validation')
parser.add_argument('--num_workers', default=24, type=int, help='num of workers')
parser.add_argument('--n_class', default=2, type=int, help='number of class')
parser.add_argument('--gpu_id', default="0", type=str, help='which gpu to run.')
parser.add_argument('--suffix', default=['.png','.jpg','.tif'], type=list, help='the suffix of the image files.')
parser.add_argument('--img_size', default=256, type=int, help='imagesize')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')


# path for loading data from folder
parser.add_argument('--hr1_train', default= file_train_path_time1, type=str, help='image at t1 in train set')
parser.add_argument('--hr2_train', default= file_train_path_time2, type=str, help='image at t2 in train set')
parser.add_argument('--lab_train', default= file_train_path_label, type=str, help='label image in train set')
# # """

parser.add_argument('--hr1_test', default= file_test_path_time1, type=str, help='image at t1 in test set')
parser.add_argument('--hr2_test', default= file_test_path_time2, type=str, help='image at t2 in test set')
parser.add_argument('--lab_test', default= file_test_path_label, type=str, help='label image in test set')

# network saving
parser.add_argument('--model_dir', default='/./', type=str, help='model save path')
