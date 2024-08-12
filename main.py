from data_loading import PoemDataset, coco_train_data, coco_test_data
import torch
import os


def main():
    poem_data = PoemDataset("poem_dataset")
    coco_train = coco_train_data
    coco_test = coco_test_data


if __name__ == '__main__':
    main()
