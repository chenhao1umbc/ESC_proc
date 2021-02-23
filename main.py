"""This file is made to process ESC-10 from the ESC-50 folder
to get the data ready for weakly supervised learning
"""
#%% load dependency 
from util import *

# %% process data
files = esc10_files.split('\n    ')[1:-1]  # splict the file names into a list
route = '/home/chenhao1/Matlab/data_matlab/ESC-50/audio/'
data = np.random.rand(400, 220500)
for i, f in enumerate(files):
    rate, d = readwav(route+f)
    data[i] = (d / np.linalg.norm(d)).squeeze()  # d is 1-d time series

f_len, t_len = 100, 100
data_ds = np.random.rand(400, f_len, t_len)  # downsampled data
for i in range(400):
    data_ds[i] = downsample(data[i], t_len=t_len, f_len=f_len)
    data_ds[i] = (data_ds[i]/ np.linalg.norm(data_ds[i]))

"get labels"
Y_all = np.zeros((400, 10))
for i in range(10):
    Y_all[i*40:(i+1)*40, i] = 1.0

# %% plot data
for i in np.arange(0, 400, 40):
    plt.figure()
    plt.plot(data[i])
    sp, sx = spectro(data[i], bw=199, overlap=0.2, fs=44.1e3, showplot=True)
    plt.figure()
    plt.imshow(data_ds[i], interpolation='None', aspect='auto')
"show labels"
plt.figure()
plt.imshow(Y_all, aspect='auto', interpolation='None')

# %% generate bags of data
"""the data are generated in the following way:
suppose there are 5 slots, for each slot there could one non-mixture sample or not
all the 5 slots must contain at least one non-mixture sample

for the Not concatanated
5 slots are stored in a list of data

for concatanated data in a bag
5 slots are stored as a tensor
"""
readme = """X is X_bag concatinated over time, yy contains per instance label, 
each bag has 5 instances, at least one instance is not empty(empty means all 0 data, no label). 
There could be several instances in one bag have the same label"""
x_tr, y_tr , x_te, y_te = split_data(data_ds, Y_all, n_test=50)
n = 800 # n is number of bags
X_bag_tr, X_tr, Y_tr, y_detail_tr = pack_data_label((x_tr, y_tr), n, f_len, t_len)

n = 200 # n is number of bags
X_bag_te, X_te, Y_te, y_detail_te = pack_data_label((x_te, y_te), n, f_len, t_len)

# sio.savemat('esc10_tr.mat',{'X':X_tr, 'X_bag':X_bag_tr, 'Y':Y_tr, 'yy':y_detail_tr, 'readme':readme})
# sio.savemat('esc10_te.mat',{'X':X_te, 'X_bag':X_bag_te, 'Y':Y_te, 'yy':y_detail_te, 'readme':readme})

#%% check data and label
a = 30
figure = plt.figure()
figure.set_size_inches(w=10, h=6)
plt.imshow(X_tr[a], interpolation='None', aspect='auto')
print(y_detail_tr[a])
print(Y_tr[a])

for i in range(5):
    figure = plt.figure()
    figure.set_size_inches(w=5, h=4)
    plt.imshow(X_bag_tr[a,i], interpolation='None', aspect='auto')

# %% check randomness
n = 100
a = np.random.rand(n)
for i in range(n):
    a[i] = np.round(np.random.rand(1)*6).astype('int')
plt.plot(a)
print([len(a[a==i]) for i in range(7)])  # head and tail are half probability

# %%
