#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics

import numpy as np
filename = 'rfcmodel3'
outfile = open(filename, 'wb')




#Importing featurelist_p1,p2,p3,p4,n1
filename = 'models/modelp5'
infile = open(filename,'rb') # Postive dataset1
p1 = pickle.load(infile)
filename = 'models/modelp6'
infile = open(filename,'rb') # Postive dataset2
p2 = pickle.load(infile)
filename = 'models/modelp3'
infile = open(filename,'rb') # Postive dataset3
p3 = pickle.load(infile)
filename = 'models/modelp4'
infile = open(filename,'rb') # Postive dataset4
p4 = pickle.load(infile)
filename = 'models/modelp2'
infile = open(filename,'rb') # Postive dataset4
p5 = pickle.load(infile)
filename = 'models/modelp1'
infile = open(filename,'rb') # Postive dataset4
p6 = pickle.load(infile)
filename = 'models/modeln4'
infile = open(filename,'rb') # negative dataset1
n1 = pickle.load(infile)

# print(p1.shape)
# print('p2=',p2.shape)
# print(p3.shape)
# print(p4.shape)
# print('n1=',n1.shape)

#Function to create dataframe from arrays
def createdf(px):
    featurelist = px
    featuredata = pd.DataFrame({   'feature 1':featurelist[:,0],
                        'feature 2':featurelist[:,1],
                        'feature 3':featurelist[:,2],
                        'feature 4':featurelist[:,3],
                        'feature 5':featurelist[:,4],
                        'feature 6':featurelist[:,5],
                        'feature 7': featurelist[:,6],
                        'feature 8': featurelist[:,7],
                        'feature 9': featurelist[:,8],
                        'feature 10': featurelist[:,9],
                        'feature 11': featurelist[:,10],
                        'feature 12': featurelist[:,11],
                        'feature 13': featurelist[:,12],
                        'feature 14': featurelist[:,13],})
    return featuredata#FunctFun


dfp1 = createdf(p1)
dfp2 = createdf(p2)
dfp3 = createdf(p3)
dfp4 = createdf(p4)
dfp5 = createdf(p5)
dfp6 = createdf(p6)
dfn1 = createdf(n1)


dfp5 = pd.concat([dfp2]) #Combining all the positve samples


dfp5['human'] = 1 #setting the category '1' as human
dfn1['human'] = 0 #setting the category '0' as human

# print(dfn1.count())
# print(dfp5.count())
dfp5 = dfp5.iloc[1:]
dfn1 = dfn1.iloc[1:]
dfn1 = dfn1.iloc[70000:]


print('count')
print(dfn1.count())
print(dfp5.count())
data = pd.concat([dfn1, dfp5])

#Setting the featueres. We choose use all the 14 columns as features
X=data[['feature 1', 'feature 2', 'feature 3', 'feature 4', 'feature 5', 'feature 6','feature 7', 'feature 8', 'feature 9',
        'feature 10','feature 11','feature 12','feature 13','feature 14']]  # Features



y=data['human']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)

rfc = RandomForestClassifier(n_estimators=600,max_depth=300, random_state=0, bootstrap= True, max_features='sqrt')
rfc.fit(X_train,y_train)
pickle.dump(rfc, outfile)
rfc_predict = rfc.predict(X_test)
rfc_probs = rfc.predict_proba(X_test)
print(confusion_matrix(y_test,rfc_predict))
print(classification_report(y_test,rfc_predict))
print(accuracy_score(y_test, rfc_predict))

print(pd.DataFrame({'feature': list(X),'importance': rfc.feature_importances_}).                    sort_values('importance', ascending = False))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, rfc_predict))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, rfc_predict))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, rfc_predict)))



print(dfp5.head())
print(dfn1.head())
print('Score: ', rfc.score(X_train, y_train))


# In[2]:


from random import randint
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import numpy as np
import pickle,sys
from sklearn.cluster import DBSCAN

import math
from sklearn.ensemble import RandomForestClassifier
import cv2

filename = 'rfcmodel2'
infile = open(filename,'rb') # pickle.dump(scanlist,outfil# e)
rfc = pickle.load(infile)

humans = 0
scannum = -1
totalcnt = 0


#Opening a serialised object
filename = 'Pickled files/p6'  #Test TDataset used for detection and counting - Conists of 2 humans.
infile = open(filename,'rb') # pickle.dump(scanlist,outfile)
scanlist = pickle.load(infile)
scan_shape = scanlist.shape
x_array = np.array([])
y_array = np.array([])


#-Data forming - End

colors = []
for x in range(20):
    colors.append('%06X' % randint(0, 0xFFFFFF))


scanlist_flat = scanlist.flatten()


def newscan(scannum):
    print('scan number : ',scannum)
    print('total count =', totalcnt)
    if(scannum <= scan_shape[0]):
        return scanlist[scannum]



class MyWidget(pg.GraphicsWindow):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.mainLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(15) # in milliseconds
        self.timer.start()
        self.timer.timeout.connect(self.onNewData)

        self.plotItem = self.addPlot(title='HUMAN DETECTION AND COUNTING \U0001f600')
        self.plotItem.getViewBox().setRange(xRange=(-3000,6000),yRange=(-6000,4000))

        self.plotDataItem = self.plotItem.plot([], pen=None,symbolBrush=(255,0,0), symbolSize=5, symbolPen=None)



    def onNewData(self):   #Each scan is passed through

        global scannum
        self.plotItem.clear()

        global humans
        humans = 0

        scannum = scannum + 1
        if scannum==scan_shape[0]-1:
            return
        if scannum==scan_shape[0]:
            print("End of scans")
            sys.exit()
        scan=newscan(scannum)

        centers = scan
        # Compute DBSCAN
        db = DBSCAN(eps=150, min_samples=15).fit(centers)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        label_list = set(labels)

        def getFeatures(clust_x, clust_y, clustersize): #Feature extraction - 14 features

            x_mean = sum(clust_x)/clustersize
            y_mean = sum(clust_y)/clustersize

            clust_x_sorted = np.sort(clust_x)
            clust_y_sorted = np.sort(clust_y)
            x_median = np.median(clust_x_sorted)
            y_median = np.median(clust_y_sorted)

            distance = math.sqrt(x_median*x_median + y_median*y_median)

            sum_std_diff = sum_med_diff = 0
            for i in range(clustersize):
                sum_std_diff += pow(clust_x[i]-x_mean, 2) + pow(clust_y[i]-y_mean, 2)
                sum_med_diff += math.sqrt(pow(clust_x[i]-x_median, 2)+pow(clust_y[i] - y_median, 2))

            std = math.sqrt(1/(clustersize-1)*sum_std_diff)
            avg_med_dev = sum_med_diff / clustersize

            first_elem = [clust_x[0],clust_y[0]]  #81
            last_elem  = [clust_x[-1],clust_y[-1]]  #82

            prev_ind = 0
            next_ind = 0
            prev_jump = 0
            next_jump = 0
            occluded_left = 0
            occluded_right = 0

            #122

            width = math.sqrt(pow(clust_x[0]-clust_x[-1], 2) + pow(clust_y[0]-clust_y[-1],2))  #125 - Width

            points=np.array((clustersize,2))

            points2=np.vstack((clust_x,clust_y))
            points2=np.transpose(points2)


            W = np.zeros((2,2), np.float64)
            w = np.zeros((2,2), np.float64)
            U = np.zeros((clustersize, 2), np.float64)

            V = np.zeros((2,2), np.float64)

            w,u,vt=cv2.SVDecomp(points2,W,U,V)
            rot_points = np.zeros((clustersize,2), np.float64)

            W[0,0]=w[0]
            W[1,1]=w[1]
            rot_points = np.matmul(u,W)


            linearity=0
            for i in range(clustersize):
                linearity += pow(rot_points[i, 1], 2)


            #Circularity
            A = np.zeros((clustersize,3), np.float64)
            B = np.zeros((clustersize,1), np.float64)


            for i in range(clustersize):
                A[i,0]=-2.0 * clust_x[i]
                A[i,1]=-2.0 * clust_y[i]
                A[i,2]=1
                B[i,0]=math.pow(clust_x[i], 2)-math.pow(clust_y[i], 2)

            sol = np.zeros((3,1),np.float64)
            cv2.solve(A, B, sol, cv2.DECOMP_SVD)

            xc = sol[0,0]
            yc = sol[1,0]
            rc = math.sqrt(pow(xc, 2)+pow(yc, 2)) - sol[2,0]


            circularity = 0
            for i in range(clustersize):
                circularity += pow(rc - math.sqrt(pow(xc - clust_x[i], 2) + pow(yc-clust_y[i], 2)), 2)


            radius = rc #Radius

            mean_curvature = 0  #207 Mean_Curvature

            boundary_length = 0  #Boundary_Length
            last_boundary_seg = 0 #Boundary_Length
            boundary_regularity = 0
            sum_boundary_reg_sq = 0

            #Mean Angualar Difference

            left = 2
            mid = 1
            right=0

            ang_diff=0

            while(left!=clustersize):
                mlx  =  clust_x[left] - clust_x[mid]
                mly  =  clust_y[left] - clust_y[mid]
                L_ml =  math.sqrt(mlx*mlx + mly*mly)

                mrx  = clust_x[right] - clust_x[mid]
                mry  = clust_y[right] - clust_y[mid]
                L_mr = math.sqrt(mrx * mrx + mry * mry)

                lrx  = clust_x[left] - clust_x[right]
                lry  = clust_y[left] - clust_y[right]
                L_lr = math.sqrt(lrx * lrx + lry * lry)


                boundary_length+= L_mr
                sum_boundary_reg_sq += L_mr*L_mr
                last_boundary_seg = L_ml

                A = (mlx * mrx + mly * mry) / pow(L_mr, 2)
                B = (mlx * mry - mly * mrx) / pow(L_mr, 2)

                th = math.atan2(B,A)

                if th<0:
                    th += 2*math.pi

                ang_diff += th/clustersize

                s = 0.5 * (L_ml + L_mr + L_lr)
                area = math.sqrt(s * (s - L_ml) * (s - L_mr) * (s - L_lr))


                if th>0:
                    mean_curvature += 4 * (area) / (L_ml * L_mr * L_lr * clustersize)
                else:
                    mean_curvature -= 4 * (area) / (L_ml * L_mr * L_lr * clustersize)


                left=left+1
                mid=mid+1
                right=right+1  #While loop ends


            boundary_length += last_boundary_seg
            sum_boundary_reg_sq += last_boundary_seg*last_boundary_seg



            boundary_regularity = math.sqrt((sum_boundary_reg_sq - math.pow(boundary_length, 2) / clustersize)/(clustersize - 1))



            #Mean Angular difference
            first = 0
            mid   = 1
            last  = -1


            sum_iav = 0
            sum_iav_sq = 0



            while(mid < clustersize-1):
                mlx = clust_x[first] -clust_x[mid]
                mly = clust_y[first] -clust_y[mid]

                mrx  = clust_x[last]-clust_x[mid]
                mry  = clust_y[last]-clust_y[mid]
                L_mr = math.sqrt(mrx * mrx + mry * mry)

                A = (mlx * mrx + mly * mry) / pow(L_mr, 2)
                B = (mlx * mry - mly * mrx) / pow(L_mr, 2)

                th = math.atan2(B, A)


                if(th<0):
                    th += 2 * math.pi


                sum_iav += th

                sum_iav_sq += th*th

                mid = mid+1


            iav = sum_iav/clustersize
            std_iav = math.sqrt((sum_iav_sq - pow(sum_iav, 2) / clustersize) / (clustersize - 1))




            features=[clustersize, std, avg_med_dev, width, linearity, circularity,
                                radius, boundary_regularity, mean_curvature, ang_diff, iav, std_iav,
                                distance, distance/clustersize]
            return features

        for label in label_list:  #Looping through the clusters in each scan
            index = labels == label
            cluster = scan[index]

            clus_max_x = max(cluster[:, 0]) + 200
            clus_min_x = min(cluster[:, 0]) - 200
            clus_max_y = max(cluster[:, 1]) + 200
            clus_min_y = min(cluster[:, 1]) - 200
            global totalcnt

            features = getFeatures(cluster[:, 0], cluster[:, 1], cluster.shape[0])

            c1 = self.plotItem.plot(cluster[:, 0], cluster[:, 1], symbol='o', symbolPen=colors[label], symbolSize=8)
            txt = '\U0001f600' + '-' + str(humans)
            txt1 = 'total humans detected =' + str(totalcnt)
            # text = pg.TextItem(html=txt, anchor=(0, 0), border='w', fill=(0, 0, 255, 100))
            label1 = pg.TextItem(html=txt1, anchor=(0, 0), border='w', fill=None)

            self.plotItem.addItem(label1)

            label1.setPos(0,2000)   #Display count label

            if (rfc.predict([features])==1) and features[0]>25 and features[4]< 700000:  #If cluster is human

                humans += 1

                temp = humans

                totalcnt = max(temp,humans) #Keeps track of the total human count detected

                txt = '\U0001f600' + '=' + str(humans) #Adding emoji and identification number

                text = pg.TextItem(html=txt, anchor=(0, 0), border='w', fill=None)
                print(features)
                clus_max_x = max(cluster[:, 0])
                clus_min_x = min(cluster[:, 0])
                clus_max_y = max(cluster[:, 1])
                clus_min_y = min(cluster[:, 1])
                x1 = [clus_min_x ,clus_min_x, clus_max_x, clus_max_x, clus_min_x]
                y1 = [clus_min_y ,clus_max_y, clus_max_y, clus_min_y, clus_min_y]
                self.plotItem.plot(x1, y1, pen='r')
                self.plotItem.addItem(text)
                text.setPos(clus_max_x, clus_max_y)

def main():
    app = QtWidgets.QApplication([])

    pg.setConfigOptions(antialias=False) # True seems to work as well
    pg.setConfigOption('background', 'w')

    win = MyWidget()
    win.show()
    win.resize(800,600)
    win.raise_()
    app.exec_()



if __name__ == "__main__":
    main()


# In[14]:


import pandas as pd
import numpy as np
import pickle
import os
def cls():
    os.system('cls' if os.name=='nt' else 'clear')

k = 0

filepath="ap5.csv"  #Change the name of the dataset here to serialise
def parsecoord(coortxt):
    coortxt=coortxt[1:-2]
    xy=coortxt.split(';')
    x=float(xy[0])
    y=float(xy[1])
    return x,y
with open(filepath,"r") as file:
    line=file.readline()
    print(k)
    scanlist=np.array([]).reshape(0,1080,2)
    while line:
        ll=line.split(',')
        ll=ll[1:-1]
        nump=len(ll)
#         print(nump) 1080
        scan=np.array([]).reshape(0,2)
        for i in range(nump):

            coortex=ll[i]
            x,y=parsecoord(coortex)
            xy=[[x,y]]
            scan=np.append(scan,xy,axis=0)
        line=file.readline()
        scan=np.reshape(scan,(1,1080,2))
        scanlist=np.append(scanlist,scan,axis=0)
        print('1')
        print('2.....')
        print('3........')
        cls()
        k = k + 1

filename = 'p5'  #Name of the seraialised object
outfile = open(filename,'wb')
pickle.dump(scanlist,outfile)
print('Data prepare complete')

print(scanlist.shape)


# In[7]:


print(type(dfp5))


# In[13]:


dfp5.to_csv("ap5.csv")


# In[15]:


dfp5


# In[ ]:




