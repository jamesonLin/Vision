import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np

digits = datasets.load_digits()

def split(data):
    size = int(len(data.target)/3)
    newDataX, newTestDataX, newDataY, newTestDataY = train_test_split(data.data, data.target, test_size=size)
    
    return newDataX, newTestDataX, newDataY, newTestDataY


def per(data, target, clf):
    pre = clf.predict(data)
    percentage = 100/len(data)
    count = 0
    for i in range(len(data)):
        if target[i] == pre[i]:
            count += 1
    return count*percentage


data1X, data2X, data1Y, data2Y = split(digits)

data1clf = svm.SVC(gamma=0.001, C=100)
data1clf.fit(data1X, data1Y)

data2clf = svm.SVC(gamma=0.001, C=100)
data2clf.fit(data2X, data2Y)


accuracy1 = per(data1X, data1Y, data2clf)
accuracy2 = per(data2X, data2Y, data1clf)




print("test")
print(len(data2X))
print(accuracy2)
print("train") 
print(len(data1X))
print(accuracy1)




    



# fig, ax = plt.subplots(ncols = 2, sharex=True, sharey=True)

# print(data1clf.predict(data1X)) #Beetje flauw, maar hij wil perse 2d-array hebben ook als er maar 1 element inzit
# ax[0].imshow(data1X, cmap=plt.cm.gray_r, interpolation='nearest')
# ax[0].set_title("trainingsdata")
# print(data2clf.predict(data2X)) #Beetje flauw, maar hij wil perse 2d-array hebben ook als er maar 1 element inzit
# ax[1].imshow(data2X, cmap=plt.cm.gray_r, interpolation='nearest')
# ax[1].set_title("testdata")
# plt.show()