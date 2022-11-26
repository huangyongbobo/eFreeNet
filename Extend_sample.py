import os
import shutil
import glob
import numpy as np


# get the max-value and min-value of target in training set
def dataset_max_min(dataset_list, target_path):
    target_max = 0
    target_min = float('inf')
    for i in range(len(dataset_list)):
        file_path, filename = os.path.split(dataset_list[i])
        groundtruth = np.load(os.path.join(target_path, filename.replace('.jpg', '.npy')))
        gt_count = np.sum(groundtruth)
        if gt_count > target_max:
            target_max = gt_count
        if gt_count < target_min:
            target_min = gt_count
    print(target_max)
    print(target_min)
    return target_max, target_min


# divide the interval and count the number of images in each interval
def dataset_range(dataset_list, target_path, target_max, target_min):
    range_num = np.zeros(10)
    total_range_size = target_max - target_min
    for i in range(len(dataset_list)):
        file_path, filename = os.path.split(dataset_list[i])
        groundtruth = np.load(os.path.join(target_path, filename.replace('.jpg', '.npy')))
        gt_count = np.sum(groundtruth)
        range_index = int(((gt_count - target_min) / total_range_size) * 10)
        # the max-target is placed in the last interval
        if gt_count == target_max:
            range_num[range_index - 1] = range_num[range_index - 1] + 1
        else:
            range_num[range_index] = range_num[range_index] + 1

    return range_num


# extend the images in training set
def extend_sample(dataset_list, target_path, target_max, target_min, range_num):
    total_range_size = target_max - target_min
    for i in range(len(dataset_list)):
        file_path, filename = os.path.split(dataset_list[i])
        groundtruth = np.load(os.path.join(target_path, filename.replace('.jpg', '.npy')))
        gt_count = np.sum(groundtruth)
        range_index = int(((gt_count - target_min) / total_range_size) * 10)
        if gt_count == target_max:
            range_index = range_index - 1
        if range_num[range_index] < int(len(dataset_list) * 0.05):
            for index in range(3):
                image_copy_path = dataset_list[i].replace('.jpg', '_' + str(index + 1) + '.jpg')
                shutil.copy2(dataset_list[i], image_copy_path)
                target_copy_path = os.path.join(target_path, filename.replace('.jpg', '_' + str(index + 1) + '.npy'))
                np.save(target_copy_path, groundtruth)


def main():
    image_path = '.../RSOC_building/train_data/images'
    # image_path = '.../VisDrone-People/train/images'
    # image_path = '.../VisDrone-Vehicle/train/images'
    target_path = '.../RSOC_building/train_data/target_center'
    # target_path = '.../VisDrone-People/train/Ground_Truth'
    # target_path = '.../VisDrone-Vehicle/train/Ground_Truth'

    train_list = []
    for img_path in glob.glob(os.path.join(image_path, '*.jpg')):
        train_list.append(img_path)

    target_max, target_min = dataset_max_min(train_list, target_path)
    range_num = dataset_range(train_list, target_path, target_max, target_min)
    extend_sample(train_list, target_path, target_max, target_min, range_num)


if __name__ == '__main__':
    main()
