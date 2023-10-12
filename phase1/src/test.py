from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from ssd import build_ssd
import cv2 as cv

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='../weights/ssd300_COCO_10000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='../results/eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.4, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def show_detection(image, bndnox, text):
    """在每个检测到的人脸上绘制一个矩形进行标示"""
    if type(bndnox) == tuple or type(bndnox) == list:
        x = int(bndnox[0])
        y = int(bndnox[1])
        w = int(bndnox[2])
        h = int(bndnox[3])
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
    else:
        for (x, y, w, h) in bndnox:
            cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
    cv.putText(image, str(text), (x + 10, y + 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image

def visualize_set(save_folder, testset):
    from matplotlib import pyplot as plt

    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        for bndbox in annotation:
            img = show_detection(img, bndbox, "")
        plt.imshow(img)
        plt.show()



def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    from matplotlib import pyplot as plt

    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        filename = save_folder + "{}".format(img_id[1]) + ".txt"
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.1:
                score = detections[0, i, j, 0].item()
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                if j == 0:
                    with open(filename, mode='w') as f:
                        f.write(label_name + ' ' +
                                str(score) + ' ' + str(coords[0]) + ' ' + str(coords[1])+' ' + str(coords[2])+' ' + str(coords[3]) + '\n')
                else:
                    with open(filename, mode='a') as f:
                        f.write(label_name + ' ' +
                                str(score) + ' ' + str(coords[0]) + ' ' + str(coords[1])+' ' + str(coords[2])+' ' + str(coords[3]) + '\n')
                img = show_detection(img, coords, str(score)[0:4])
                j += 1
        # if you want to visualize the detection results
        # plt.imshow(img)
        # plt.show()



def test_voc():
    # load net
    num_classes = len(VOC_CLASSES)+1
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = VOCDetection(args.voc_root, 'test', None, VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    # visualize_set(args.save_folder, testset)
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test_voc()
