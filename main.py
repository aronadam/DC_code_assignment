import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as im
from scipy import fftpack as ff

#load input image
impath = "MESSI_image.jpeg"
img = im.open(impath)

#extract data into workable format
img_data = np.array(img)

# create black & white image
img_data_bw = img_data[:,:,1]
img_bw = im.fromarray(np.uint8(img_data))
# plt.imshow(bw_data,cmap='Greys_r')
# plt.show()

# $TODO--> fix minimizer, get sequence out of input..?
def l1_min(y,A,max_iter):
    p_0 = np.zeros(np.shape(y))
    z_0 = np.zeros(np.shape(y))
    epsilon = 1

    p = np.zeros([max_iter,1])
    p[0] = p_0
    z = np.zeros([max_iter,1])
    z[0] = z_0
    i=0
    eta=.01
    while epsilon<1e-4 and i<max_iter:
        p[i+1] = z[i]-eta*l1_grad(z[i])
        z[i+1] = p[i+1]-A.T*(np.invert((A.T*A)))*(A*p[i+1]*y)
        i=+1
    return z[i]

def l1_grad(x):
    if x>0:
        return 1
    elif x<0:
        return -1
    else:
        return np.random.random()

def mse(v1,v2):
    if v1.shape != v2.shape:
        return print("matrices must have equal size.")
    else:
        return np.sum((v1-v2)**2)/(v1.shape[0]*v1.shape[1])

#kernel parameter
s = 32

p_range = int(len(img_data)/s)

k_sparse = np.zeros(np.shape(img_data_bw))
unit_basis = np.eye(s,s)

mse_unit = np.zeros([p_range*p_range,1])
mse_dct = mse_unit.copy()


for i in range(p_range):
    for j in range(p_range):
        if np.mod(i,2)==0:
            j_hat = j
        else:
            j_hat = (p_range-1)-j
        patch = img_data_bw[s*i:s*(i+1),s*j_hat:s*(j_hat+1)]
        sparsity = s*s-np.count_nonzero(patch==0)
        print(sparsity)

        dct_basis = ff.dct(patch)
        
        s_dct = dct_basis*patch
        s_unit = unit_basis*patch

        mse_dct[4*i+j] = np.log10(mse(patch,s_dct))
        mse_unit[4*i+j] = np.log10(mse(patch,s_unit))

        # z_dct = l1_min(patch,dct_basis)
        # z_unit = l1_min(patch,unit_basis)
        # $TODO-->fix MSE
#slide 11!!!
#L1 norm minimization
plt.plot(mse_dct,'r')
plt.plot(mse_unit,'b')
plt.legend(["DCT","Unit"])
plt.title("MSE (dB) for each image segment")
plt.show()


######### part B