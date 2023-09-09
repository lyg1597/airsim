import os 
import json 
import matplotlib.pyplot as plt 
import scipy.optimize
import scipy.ndimage
import numpy as np 

dir = './picture_sc/data'

W_list = []
dhat_list = []
for idx in range(10000):
    fn = os.path.join(dir, f'{idx}.json')
    with open(fn, 'r') as f:
        res = json.load(f)
    if res['detect_flag']:
        dhat = abs(res['s_star'][1])
        W = res['W']
        W_list.append(W)
        dhat_list.append(dhat)

idx = np.argsort(W_list)

W_list = np.array(W_list)[idx]
dhat_list = np.array(dhat_list)[idx]
dhat_list_filtered = scipy.ndimage.filters.median_filter(np.array([dhat_list,W_list]), size = 8)[0,:]
# dhat_list_filtered = dhat_list 
plt.plot(W_list, dhat_list_filtered,'.')
plt.show()
res, _ = scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  W_list,  dhat_list_filtered)
print(res)
K1 = res[0]
K2 = res[1]
K3 = 0
x = np.arange(0,1,0.01)
y = K1*np.exp(K2*x)+K3
plt.plot(W_list, dhat_list_filtered,'.')
plt.plot(x,y)
plt.show()