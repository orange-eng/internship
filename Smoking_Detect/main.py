import argparse
import os
from ksong_smoking_check_gpu_cpu import Detect
import torch
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="Testing Smoking datasets...")
parser.add_argument('--weights', nargs='+', type=str, default=r'runs/evolution/weights/best.pt',
                    help='model.pt path(s)')
parser.add_argument('--source', type=str, default='images1',
                    help='source')  # file/folder, 0 for webcam
parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.85, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--classes', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', default=True, help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
opt = parser.parse_args()
print(opt)


if __name__ == "__main__":
    results = []
    results_path = 'smoking_results.csv'

    smoking_annotations = pd.read_excel(r'smoking_label.xlsx')  # 存放ugcid和label的文件路径
    ugcids, labels = smoking_annotations['ugcid'], smoking_annotations['label']
    ugcid_values = ugcids.values

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        detector = Detect(opt)
        for path in os.listdir(opt.source):
            result = detector.detect_v2(os.path.join(opt.source, path))  # detect whether it is smoking
            print(path, 'smoking:', result)
            results.append([path, result])
            if path in ugcid_values:
                label = labels[ugcids[ugcid_values == path].index[0]]
                if result == 1:
                    if result == label:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if result == label:
                        tn += 1
                    else:
                        fn += 1
            else:
                continue

    matrix = np.array([
        [tp, fp],
        [fn, tn],
    ])
    print(matrix)
    print('precision = {}\n'.format(tp / (tp + fp)))
    print('recall = {}\n'.format(tp / (tp + fn)))
    print('acc = {}\n'.format((tp+tn) / (tp+tn+fp+fn)))

    results = pd.DataFrame(columns=['ugcid', 'result'], data=results)
    results.to_csv(results_path, index=False)

