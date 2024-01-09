#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import math
import onnxruntime as ort
import time

CLASSES = ['person']

ObjThresh = 0.6
input_imgH = 640
input_imgW = 640
max_num = 100


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def precess_image(img_src, resize_w, resize_h):
    image = cv2.resize(img_src, (resize_w, resize_h))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image /= 255

    image -= (0.485, 0.456, 0.406)
    image /= (0.229, 0.224, 0.225)
    return image


def postprocess(pred_results, img_h, img_w):

    output = []
    for i in range(len(pred_results)):
        output.append(pred_results[i].reshape((-1)))

    pred_logits = output[0]
    pred_boxes = output[1]

    print(len(pred_logits))
    print(len(pred_boxes))

    predBoxs = []
    for i in range(max_num):
        softmaxsum = 0
        for c in range(len(CLASSES) + 2):
            pred_logits[i * (len(CLASSES) + 2) + c] = math.exp(pred_logits[i * (len(CLASSES) + 2) + c])
            softmaxsum += pred_logits[i * (len(CLASSES) + 2) + c]

        for c in range(len(CLASSES) + 2):
            pred_logits[i * (len(CLASSES) + 2) + c] /= softmaxsum

        softmaxmax = 0
        softmaxindex = 0
        for c in range(len(CLASSES) + 2):
            if c == 0:
                softmaxmax = pred_logits[i * (len(CLASSES) + 2) + c]
                softmaxindex = c
            else:
                if softmaxmax < pred_logits[i * (len(CLASSES) + 2) + c]:
                    softmaxmax = pred_logits[i * (len(CLASSES) + 2) + c]
                    softmaxindex = c

        if softmaxmax > ObjThresh and softmaxindex == 1:
            x_c, y_c, w, h = pred_boxes[i * 4 + 0], pred_boxes[i * 4 + 1], pred_boxes[i * 4 + 2], pred_boxes[i * 4 + 3]
            box = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]

            rect = DetectBox(softmaxindex, softmaxmax, box[0] * img_w, box[1] * img_h, box[2] * img_w, box[3] * img_h)
            predBoxs.append(rect)

    return predBoxs


def detect(img_path):
    orig = cv2.imread(img_path)
    img_h, img_w = orig.shape[:2]
    image = precess_image(orig, input_imgW, input_imgH)

    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)

    ort_session = ort.InferenceSession('./detr_r50_person_zq_sim.onnx')

    t1 = time.time()
    for i in range(1000):
        pred_results = (ort_session.run(None, {'data': image}))
    t2 = time.time()
    print('run one time :', (t2 - t1) / 1000.0)
    print('pred_results:', len(pred_results))

    for i in range(len(pred_results)):
        print(pred_results[i].shape)

    predbox = postprocess(pred_results, img_h, img_w)

    print('obj num is :', len(predbox))

    for i in range(len(predbox)):
        xmin = int(predbox[i].xmin + 0.5)
        ymin = int(predbox[i].ymin + 0.5)
        xmax = int(predbox[i].xmax + 0.5)
        ymax = int(predbox[i].ymax + 0.5)
        classId = predbox[i].classId
        score = predbox[i].score

        cv2.rectangle(orig, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        ptext = (xmin, ymin + 15)
        title = str(CLASSES[classId - 1]) + ":%.2f" % score
        print(classId)
        cv2.putText(orig, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite('./test_onnx_result.jpg', orig)


if __name__ == '__main__':
    print('This is main ....')
    img_path = './test.jpg'
    detect(img_path)
