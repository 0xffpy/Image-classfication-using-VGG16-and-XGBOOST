import xgboost as xgb
from sklearn.metrics import confusion_matrix
import seaborn as sns
import keras
import os
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics


def x_y_train_test(parent, start_ind, end_ind):
    list_of_shapes = os.listdir(parent)
    print(list_of_shapes)
    images_list = []
    labels = []

    for i in list_of_shapes:
        paths = os.path.join(parent, i)
        shape_list = os.listdir(paths)
        for j in range(start_ind, end_ind):
            img = cv2.imread(os.path.join(paths,shape_list[j]))
            img = cv2.resize(img, (150, 150))
            images_list.append(img/255.0)
            labels.append(i)
        print(i)


    x_train, x_test, y_train, y_test = train_test_split(
                                                        images_list,
                                                        labels,
                                                        test_size=0.25,
                                                        random_state=42)
    images_list = None
    labels = None
    
    return np.array(x_train), np.array(x_test), np.array(y_train) ,np.array(y_test)

def main():
    modeo_base = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(150,150,3))

    for layer in modeo_base.layers:
        layer.trainable = False

    model = xgb.XGBClassifier()
    parent = "dataset_file/train"

    for i in range(8):
        x_train, x_test, y_train, y_test = x_y_train_test(parent, 1000*i, 1000+1000*i)
        labe = LabelEncoder()
        labe.fit(y_train)
        y_train = labe.transform(y_train)
        y_test = labe.transform(y_test)
        feature_extractor = modeo_base.predict(x_train)
        x_train = None
        features = feature_extractor.reshape(feature_extractor.shape[0],-1)
        model.fit(features, y_train, model= model)
        features = None
        y_train = None
        x_features_test = modeo_base.predict(x_test)
        x_features_test = x_features_test.reshape(x_features_test.shape[0],-1)
        x_test = None
        prediction = model.predict(x_features_test)
        prediction = labe.inverse_transform(prediction)
        x_features_test = None
        print ("Accuracy = ", metrics.accuracy_score(labe.inverse_transform(y_test), prediction))
        cm = confusion_matrix(labe.inverse_transform(y_test),prediction)
        sns.heatmap(cm, annot=True)

if __name__ == "__main__":
        main()
