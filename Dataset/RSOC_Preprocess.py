import h5py
import scipy.io as io
import numpy as np
import os
import glob
from matplotlib import pyplot as plt


def main():
    root = '/media/ysliu/6b94d4ca-f5c4-46ae-8497-af46d2544dfc/Maoer/RSOC_building'
    building_train = os.path.join(root, 'train_data', 'images')
    building_test = os.path.join(root, 'test_data', 'images')

    path_sets = [building_train, building_test]

    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    for img_path in img_paths:
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
        img = plt.imread(img_path)
        k = np.zeros((img.shape[0], img.shape[1]))
        gt = mat['center'][0, 0]
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1

        target = img_path.replace('.jpg', '.npy').replace('images', 'target_center')
        np.save(target, k)


if __name__ == '__main__':
    main()
