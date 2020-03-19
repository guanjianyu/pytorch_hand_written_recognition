import os
import cv2
import glob
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import torch
from torch import nn
import torchvision
import torch.nn.functional as F


def get_char_dict():
    char_dict = {}
    char_dict["pad"] = 0
    char_dict["sos"] = 1
    char_dict["eos"] = 2
    for i in range(32, 127):
        char_dict[chr(i)] = len(char_dict)

    inverse_char_dict = {v: k for k, v in char_dict.items()}
    return char_dict, inverse_char_dict


def resize_image(image, desired_size):
    ''' Helper function to resize an image while keeping the aspect ratio.
    Parameter
    ---------

    image: np.array
        The image to be resized.
    desired_size: (int, int)
        The (height, width) of the resized image
    Return
    ------
    image: np.array
        The image of size = desired_size
    bounding box: (int, int, int, int)
        (x, y, w, h) in percentages of the resized image of the original
    '''
    size = image.shape[:2]
    if size[0] > desired_size[0] or size[1] > desired_size[1]:
        ratio_w = float(desired_size[0]) / size[0]
        ratio_h = float(desired_size[1]) / size[1]
        ratio = min(ratio_w, ratio_h)
        new_size = tuple([int(x * ratio) for x in size])
        image = cv2.resize(image, (new_size[1], new_size[0]))
        size = image.shape

    delta_w = max(0, desired_size[1] - size[1])
    delta_h = max(0, desired_size[0] - size[0])

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = image[0][0]
    if color < 230:
        color = 230
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=float(color))
    crop_bb = (left / image.shape[1], top / image.shape[0], (image.shape[1] - right - left) / image.shape[1],
               (image.shape[0] - bottom - top) / image.shape[0])
    image[image > 230] = 255
    return image, crop_bb


def get_transform(phase="train"):
    transfrom_PIL_list = [
        transforms.RandomAffine((-2, 2), fillcolor=255),
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(contrast=0.5),
        transforms.ColorJitter(saturation=0.5),

    ]
    transfrom_tensor_list = [
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), value=0),
    ]
    if phase == "train":
        transform = transforms.Compose([
            transforms.RandomApply(transfrom_PIL_list),
            transforms.ToTensor(),
            # transforms.RandomApply(transfrom_tensor_list),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]),
        ])
    return transform

def read_img(img, inference_transform,desired_size=(128, 1024)):
    img_resize, crop_cc = resize_image(img, desired_size)
    img_resize = Image.fromarray(img_resize)
    img_tensor = inference_transform(img_resize)
    return img_tensor


class IAM_Dataset_line(data.Dataset):
    """
    pytorch dataset for IAM handwritten dataset
    """
    def __init__(self, dataset_path, tokenizer, phase="train", padding=128):
        """
        inital dataset with path, tokenizer, when you want to train phase is "train",
        for test and valid phase is "test" and "valid", padding is the max length of
        your input text.
        """
        self.phase = phase
        self.dataset_path = dataset_path
        self.padding = padding
        self.tokenizer = tokenizer

        self.line_imgs = self.read_train_valid_test_files()
        self.label_dict = self.read_label_dict()
        self.transform = get_transform(self.phase)

    def __len__(self):
        return len(self.line_imgs)

    def read_label_dict(self):
        """
        Read the line ground truth data from txt as dict
        key is file name, values is texts
        """
        line_txt_path = os.path.join(self.dataset_path, "ascii/lines.txt")
        with open(line_txt_path, "r") as f:
            lines = f.readlines()

        ground_truth_dict = {}
        for line in lines:
            if line.startswith("#"):
                continue
            line = line.strip()
            line_blocks = line.split(" ")
            key = line_blocks[0]
            texts = line_blocks[-1].replace("|", " ")
            ground_truth_dict[key] = texts
        return ground_truth_dict

    def read_train_valid_test_files(self):
        """
        Split all line imgs into train,valid,test set.
        These sets are divided based on file level, which means line imgs from same file
        will not be divided into different set
        """

        np.random.seed(55)
        folder_path = os.path.join(self.dataset_path, "lines")
        folders = glob.glob(os.path.join(folder_path, "*"))

        files = []
        for folder in folders:
            files_in_folder = glob.glob(os.path.join(folder, "*"))
            files.extend(files_in_folder)

        train_file_num = int(len(files) * 0.9)
        valid_file_num = int(len(files) * 0.05)

        files_permute = np.random.permutation(files)

        train_files = files_permute[:train_file_num]
        valid_files = files_permute[train_file_num:train_file_num + valid_file_num]
        test_files = files_permute[train_file_num + valid_file_num:]

        train_lines = []
        valid_lines = []
        test_lines = []
        files_tuple = [(train_lines, train_files), (valid_lines, valid_files), (test_lines, test_files)]

        for phase_lines, phase_files in files_tuple:
            for file_folder in phase_files:
                file_imgs = glob.glob(os.path.join(file_folder, "*.png"))
                for img_path in file_imgs:
                    phase_lines.append((img_path, os.path.basename(img_path).split(".")[0]))

        print("Total files: ", len(files))
        print("Train files: ", len(train_files))
        print("Valid files: ", len(valid_files))
        print("Test files: ", len(test_files))

        if self.phase == "train":
            return train_lines
        elif self.phase == "valid":
            return valid_lines
        else:
            return test_lines

    def read_img(self, img_path, desired_size=(128, 1024)):
        img = cv2.imread(img_path, 0)
        img_resize, crop_cc = resize_image(img, desired_size)
        img_resize = Image.fromarray(img_resize)
        img_tensor = self.transform(img_resize)
        return img_tensor

    def read_label(self, label_key):

        text = self.label_dict[label_key]
        line_label = self.tokenizer.encode(text)
        input_lengths = len(line_label)
        if self.padding > 0:
            padded_line = np.ones(self.padding)*-1
            max_len = min(self.padding, input_lengths)
            padded_line[:max_len] = line_label[:max_len]
            line_label = padded_line
            input_lengths = max_len
        label_tensor = torch.from_numpy(line_label)
        input_lengths = torch.tensor(input_lengths)
        return label_tensor, input_lengths

    def __getitem__(self, index):
        image_path, label_key = self.line_imgs[index]
        X = self.read_img(image_path)
        y, lengths = self.read_label(label_key)
        return X.float(), y.long(), lengths.long()



