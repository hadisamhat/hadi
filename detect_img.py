# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np 
import time

setup_gpu(1)

class Predict:
    def __init__(self,model_path,labels):
        self.model = models.load_model(model_path, backbone_name='resnet50')
        self.labels = labels


    def predict(self,image_path):
        image = read_image_bgr(image_path)

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        start = time.time()
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        # correct for image scale
        boxes /= scale

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break
                
            color = label_color(label)
            
            b = box.astype(int)
            draw_box(draw, b, color=color)
            
            caption = "{} {:.3f}".format(self.labels[label], score)
            draw_caption(draw, b, caption)
            return draw
        # plt.figure(figsize=(15, 15))
        # plt.axis('off')
        # plt.imshow(draw)
        # plt.show()    
labels = {0:'person'}
pred = Predict('snapshots/resnet50_csv_17_inference.h5',labels)

draw  = pred.predict('person.jpg')
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()    
