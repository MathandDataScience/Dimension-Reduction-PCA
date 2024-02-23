from astropy.io import fits
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import math

#Finds the data that we want to look at
img_idx = cv2.imread("index1.tif", cv2.IMREAD_GRAYSCALE)
col=[]
row=[]
for i in range(len(img_idx)):
    for k in range(len(img_idx[1])):
        if img_idx[i,k] < 50:
            col.append(i)
            row.append(k)
col=np.array(col)
row=np.array(row)

def good_data(img):
    for vals in range(len(col)):
        img[col[vals],row[vals]] = 0
    return img


def MxM(A, M):  # inputs must be num py arrays not list of list
    if A.shape[1] == M.shape[0]:
        M1 = []
        M2 = []
        for j in range(len(A)):
            for l in range(len(A)):
                z = np.dot(A[j], M[:, l])
                M1.append(z)
            M2.append(M1)
            M1 = []
        M2 = np.asarray(M2)  # out puts np array vs list
        return M2
    else:
        print("Error: A " + str(A.shape) + " matrix cannot be muliply by " + str(M.shape))

def flat_std_img(img):
    img = good_data(img)
    img = img.flatten()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = img - img.mean() # center data
    return img
# open files and std/flatten
H, W = fits.open('f487n.fits')[1].data.shape
img_487 = flat_std_img(fits.open('f487n.fits')[1].data)
img_502 = flat_std_img(fits.open('f502n.fits')[1].data)
img_547 = flat_std_img(fits.open('f547m.fits')[1].data)
img_656 = flat_std_img(fits.open('f656n.fits')[1].data)
img_658 = flat_std_img(fits.open('f658n.fits')[1].data)
img_673 = flat_std_img(fits.open('f673n.fits')[1].data)


# make complete matrix
data = np.stack((img_487, img_502, img_547, img_656, img_658, img_673), axis=1)

# qr factorization
Q, R = np.linalg.qr(data, mode='reduced')
R_T = R.T
# covariance matrix
cov = MxM(R_T, R)

# eig decomp of cov
values, vectors = np.linalg.eig(cov)
#order by eig vals
vector_idx = np.argsort(values)

def map_back(PC,E):
    VD = PC * E
    R1 = np.dot(VD, PC.T)
    R1 = R1.dot(np.linalg.inv(R_T))
    A1 = np.dot(Q, R1)
    return A1
# get PC's
PC1 = vectors[:, vector_idx[-1]] # direction of the unit vector for PC1
E1 = values[vector_idx[-1]]
PC1 = np.row_stack(PC1)
img1 = map_back(PC1,E1)

PC2 = vectors[:, vector_idx[-2]]
E2 = values[vector_idx[-2]]
PC2 = np.row_stack(PC2)
img2 = map_back(PC2,E2)

PC3 = vectors[:, vector_idx[-3]]
E3 = values[vector_idx[-3]]
PC3 = np.row_stack(PC3)
img3 = map_back(PC3,E3)

PC4 = vectors[:, vector_idx[-4]]
E4 = values[vector_idx[-4]]
PC4 = np.row_stack(PC4)
img4 = map_back(PC4,E4)

PC5 = vectors[:, vector_idx[-5]]
E5 = values[vector_idx[-5]]
PC5 = np.row_stack(PC5)
img5 = map_back(PC5,E5)

PC6 = vectors[:, vector_idx[-6]]
E6 = values[vector_idx[-6]]
PC6 = np.row_stack(PC6)
img6 = map_back(PC6,E6)

evnames = ["E1","E2","E3","E4","E5","E6"]
ev = [E1,E2,E3,E4,E5,E6]

plt.figure("scree")
plt.bar(evnames, ev)
plt.show()

print(ev)
print(sum(ev))
print(ev[0]/sum(ev))
print((ev[0]+ev[1]+ev[2])/sum(ev))



def unflat_img2(img):
    img = (img[:, 0] + img[:, 1] + img[:, 2] + img[:, 3] + img[:, 4] + img[:, 5]) * 1.0
    img = img ** 2 # moves negatives up  vs abs
    img = np.log(img)/np.max(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img**2)
    """plt.hist(img)
    plt.show()"""
    img = img.reshape((H, W))
    return img


PC = unflat_img2(img1)
PC_1= np.zeros((H, W))
PC_1 = good_data(PC)
PC_1 = cv2.resize(PC_1, (H // 2, W // 2))
cv2.imshow("PC1", PC_1)
cv2.imwrite("PC1.tif", PC_1 )


# PC1 - 3 images
img_T = np.zeros((H, W, 3))
img1 = unflat_img2(img1)
img2 = unflat_img2(img2)
img3 = unflat_img2(img3)
img_T[:, :, 1] = good_data(img1)
img_T[:, :, 0] = good_data(img2)
img_T[:, :, 2] = good_data(img3)
img_T = cv2.resize(img_T, (H // 2, W // 2))
# PC1 - 3 images
img_B = np.zeros((H, W, 3))
img4 = unflat_img2(img4)
img5 = unflat_img2(img5)
img6 = unflat_img2(img6)
img_B[:,:,1] = good_data(img4)
img_B[:,:,0] = good_data(img5)
img_B[:,:,2] = good_data(img6)


# displays a photo
img_B = cv2.resize(img_B,(H//2,W//2))
img_T = cv2.resize(img_T,(H//2,W//2))

cv2.imshow("PC1-3", img_T)
#cv2.imwrite("PC1-3.tif", img_T)
cv2.imshow("PC4-6", img_B)
#cv2.imwrite("PC4-6.tif", img_B)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()



