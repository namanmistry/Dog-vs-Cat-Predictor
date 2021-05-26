import numpy as np
import cv2
import pickle
import os
import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.multiclass import OneVsRestClassifier
# data = []
# for file in os.listdir("./data/train"):
#     img = cv2.imread(f"./data/train/{file}")
#     if "cat" in file:
#         label = 0
#     else:
#         label = 1
#     try:
#         img = cv2.resize(img,(50,50))
#         img_array = np.array(img).flatten()
#         data.append([img_array, label])
#     except Exception as e:
#         pass

# pickle_open = open("data.pkl","rb")
# data = pickle.load(pickle_open)
# pickle_open.close()

# random.shuffle(data)
# features = []
# labels = []

# count = 0
# for feature, label in data:
#     features.append(feature)
#     labels.append(label)
#     if count >=10000:
#         break
#     count+=1

# model = OneVsRestClassifier(SVC(C=1, gamma='auto', kernel='poly'),n_jobs=-1)
# model.fit(features, labels)

# dump(model, "./model/Model.joblib")

model = load("./model/Model.joblib")
pickle_open = open("data.pkl","rb")
data = pickle.load(pickle_open)
random.shuffle(data)
test_data = data[15000:15500]
print(len(test_data))
test_x = []
test_y = []
for feature, label in test_data:
    test_x.append(feature)
    test_y.append(label)

pred = model.predict(test_x)
print(pred)
print(test_y)
print(model.score(test_x,test_y))