import os
import cv2
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import string
import numpy as np
from tensorflow.keras import backend as K
from . import dataset
from . import model
from . import tokenizer

"""
API example:
detector = HandWritten()
detector.detect(img_path = "images/22.png")
or
img_array = cv2.imread(img_path, 0)
detector.detect(img = img_array)
"""

class HandWritten:
    def __init__(self,model_path = os.path.abspath("./handwritten_recognition/checkpoints/handwritten_model_attention.pt")):
        self.inference_transform = dataset.get_transform(phase="test")
        self.charset_base = string.printable[:95]
        self.inference_tokenizer = tokenizer.Tokenizer(self.charset_base)

        self.model =  model.flor_attention()
        checkpoints = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(checkpoints["model"])
        self.model.eval()

    def ctc_decode(self,logits):
        out = np.ones((1, 128, 98)).astype(np.float32) * 0.1
        out[:, :, :-1] = logits
        input_length = len(max(logits, key=len))

        predicts, probabilities = [], []
        x_test = np.asarray(out)
        x_test_len = np.asarray([128 for _ in range(len(x_test))])

        decode, log = K.ctc_decode(x_test,
                                   x_test_len,
                                   greedy=False,
                                   beam_width=100,
                                   top_paths=1)

        probabilities.extend([np.exp(x) for x in log])
        decode = [[[int(p) for p in x if p != -1] for x in y] for y in decode]
        predicts = np.swapaxes(decode, 0, 1)
        return predicts


    def detect(self,img=None,img_path=None):
        """
        Give gray img array or img_path, It will return detected text as string
        :param img: gray scale image array (W,H)
        :param img_path: img path
        :return: text string
        """
        if img_path is not None:
            img = cv2.imread(img_path, 0)

        if img is not None:
            img_tensor = dataset.read_img(img, self.inference_transform, desired_size=(128, 1024))
            logits = self.model(img_tensor.unsqueeze(0)).squeeze(-1).softmax(-1).data.cpu().numpy()

            pred = self.ctc_decode(logits)
            pred_sen = self.inference_tokenizer.decode(pred[0][0])
            return pred_sen
        return ""