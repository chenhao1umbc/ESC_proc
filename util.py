#This file contains all the packages and functions for this project
import datetime
import wave
import cv2

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import scipy.io as sio
from sklearn import metrics

tt = datetime.datetime.now
# torch.set_default_dtype(torch.double)
np.set_printoptions(linewidth=160)
torch.set_printoptions(linewidth=160)
torch.backends.cudnn.deterministic = True
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


#%%  
esc10_files = """
    1-116765-A-41.wav
    1-19898-A-41.wav
    1-19898-B-41.wav
    1-19898-C-41.wav
    1-47250-A-41.wav
    1-47250-B-41.wav
    1-64398-A-41.wav
    1-64398-B-41.wav
    2-50667-A-41.wav
    2-50667-B-41.wav
    2-50668-A-41.wav
    2-50668-B-41.wav
    2-68391-A-41.wav
    2-68391-B-41.wav
    2-77945-A-41.wav
    2-77945-B-41.wav
    3-118656-A-41.wav
    3-118657-A-41.wav
    3-118657-B-41.wav
    3-118658-A-41.wav
    3-118658-B-41.wav
    3-118972-A-41.wav
    3-118972-B-41.wav
    3-165856-A-41.wav
    4-149294-A-41.wav
    4-149294-B-41.wav
    4-157611-A-41.wav
    4-157611-B-41.wav
    4-165823-A-41.wav
    4-165823-B-41.wav
    4-169127-A-41.wav
    4-169127-B-41.wav
    5-170338-A-41.wav
    5-170338-B-41.wav
    5-171653-A-41.wav
    5-185579-A-41.wav
    5-185579-B-41.wav
    5-216370-A-41.wav
    5-216370-B-41.wav
    5-222524-A-41.wav
    1-21934-A-38.wav
    1-21935-A-38.wav
    1-35687-A-38.wav
    1-42139-A-38.wav
    1-48413-A-38.wav
    1-57163-A-38.wav
    1-62849-A-38.wav
    1-62850-A-38.wav
    2-119748-A-38.wav
    2-127108-A-38.wav
    2-131943-A-38.wav
    2-134700-A-38.wav
    2-135728-A-38.wav
    2-140147-A-38.wav
    2-141584-A-38.wav
    2-88724-A-38.wav
    3-142593-A-38.wav
    3-143933-A-38.wav
    3-150363-A-38.wav
    3-164688-A-38.wav
    3-170377-A-38.wav
    3-170383-A-38.wav
    3-171012-A-38.wav
    3-171041-A-38.wav
    4-175945-A-38.wav
    4-179984-A-38.wav
    4-181035-A-38.wav
    4-181865-A-38.wav
    4-188033-A-38.wav
    4-191327-A-38.wav
    4-194711-A-38.wav
    4-198965-A-38.wav
    5-201194-A-38.wav
    5-208624-A-38.wav
    5-209698-A-38.wav
    5-209833-A-38.wav
    5-210571-A-38.wav
    5-212181-A-38.wav
    5-219342-A-38.wav
    5-235671-A-38.wav
    1-17150-A-12.wav
    1-17565-A-12.wav
    1-17742-A-12.wav
    1-17808-A-12.wav
    1-17808-B-12.wav
    1-4211-A-12.wav
    1-46272-A-12.wav
    1-7057-A-12.wav
    2-18766-A-12.wav
    2-18766-B-12.wav
    2-28314-A-12.wav
    2-28314-B-12.wav
    2-30322-A-12.wav
    2-30322-B-12.wav
    2-61311-A-12.wav
    2-65747-A-12.wav
    3-104632-A-12.wav
    3-104958-A-12.wav
    3-120644-A-12.wav
    3-145774-A-12.wav
    3-147965-A-12.wav
    3-157187-A-12.wav
    3-158476-A-12.wav
    3-65748-A-12.wav
    4-164661-A-12.wav
    4-164661-B-12.wav
    4-170247-A-12.wav
    4-170247-B-12.wav
    4-171207-A-12.wav
    4-181563-A-12.wav
    4-182368-A-12.wav
    4-182369-A-12.wav
    5-186924-A-12.wav
    5-189212-A-12.wav
    5-189237-A-12.wav
    5-193473-A-12.wav
    5-193473-B-12.wav
    5-213802-A-12.wav
    5-215658-A-12.wav
    5-215658-B-12.wav
    1-187207-A-20.wav
    1-211527-A-20.wav
    1-211527-B-20.wav
    1-211527-C-20.wav
    1-22694-A-20.wav
    1-22694-B-20.wav
    1-60997-A-20.wav
    1-60997-B-20.wav
    2-107351-A-20.wav
    2-107351-B-20.wav
    2-151079-A-20.wav
    2-50665-A-20.wav
    2-50666-A-20.wav
    2-66637-A-20.wav
    2-66637-B-20.wav
    2-80482-A-20.wav
    3-151080-A-20.wav
    3-151081-A-20.wav
    3-151081-B-20.wav
    3-152007-A-20.wav
    3-152007-B-20.wav
    3-152007-C-20.wav
    3-152007-D-20.wav
    3-152007-E-20.wav
    4-167077-A-20.wav
    4-167077-B-20.wav
    4-167077-C-20.wav
    4-185575-A-20.wav
    4-185575-B-20.wav
    4-185575-C-20.wav
    4-59579-A-20.wav
    4-59579-B-20.wav
    5-151085-A-20.wav
    5-198411-A-20.wav
    5-198411-B-20.wav
    5-198411-C-20.wav
    5-198411-D-20.wav
    5-198411-E-20.wav
    5-198411-F-20.wav
    5-198411-G-20.wav
    1-100032-A-0.wav
    1-110389-A-0.wav
    1-30226-A-0.wav
    1-30344-A-0.wav
    1-32318-A-0.wav
    1-59513-A-0.wav
    1-85362-A-0.wav
    1-97392-A-0.wav
    2-114280-A-0.wav
    2-114587-A-0.wav
    2-116400-A-0.wav
    2-117271-A-0.wav
    2-118072-A-0.wav
    2-118964-A-0.wav
    2-122104-A-0.wav
    2-122104-B-0.wav
    3-136288-A-0.wav
    3-144028-A-0.wav
    3-155312-A-0.wav
    3-157695-A-0.wav
    3-163459-A-0.wav
    3-170015-A-0.wav
    3-180256-A-0.wav
    3-180977-A-0.wav
    4-182395-A-0.wav
    4-183992-A-0.wav
    4-184575-A-0.wav
    4-191687-A-0.wav
    4-192236-A-0.wav
    4-194754-A-0.wav
    4-199261-A-0.wav
    4-207124-A-0.wav
    5-203128-A-0.wav
    5-203128-B-0.wav
    5-208030-A-0.wav
    5-212454-A-0.wav
    5-213855-A-0.wav
    5-217158-A-0.wav
    5-231762-A-0.wav
    5-9032-A-0.wav
    1-172649-A-40.wav
    1-172649-B-40.wav
    1-172649-C-40.wav
    1-172649-D-40.wav
    1-172649-E-40.wav
    1-172649-F-40.wav
    1-181071-A-40.wav
    1-181071-B-40.wav
    2-188822-A-40.wav
    2-188822-B-40.wav
    2-188822-C-40.wav
    2-188822-D-40.wav
    2-37806-A-40.wav
    2-37806-B-40.wav
    2-37806-C-40.wav
    2-37806-D-40.wav
    3-150979-A-40.wav
    3-150979-B-40.wav
    3-150979-C-40.wav
    3-154926-A-40.wav
    3-154926-B-40.wav
    3-68630-A-40.wav
    3-68630-B-40.wav
    3-68630-C-40.wav
    4-125929-A-40.wav
    4-161579-A-40.wav
    4-161579-B-40.wav
    4-175000-A-40.wav
    4-175000-B-40.wav
    4-175000-C-40.wav
    4-193480-A-40.wav
    4-193480-B-40.wav
    5-177957-A-40.wav
    5-177957-B-40.wav
    5-177957-C-40.wav
    5-177957-D-40.wav
    5-177957-E-40.wav
    5-191131-A-40.wav
    5-205898-A-40.wav
    5-220955-A-40.wav
    1-17367-A-10.wav
    1-21189-A-10.wav
    1-26222-A-10.wav
    1-29561-A-10.wav
    1-50060-A-10.wav
    1-54958-A-10.wav
    1-56311-A-10.wav
    1-63871-A-10.wav
    2-101676-A-10.wav
    2-117625-A-10.wav
    2-72970-A-10.wav
    2-73027-A-10.wav
    2-73260-A-10.wav
    2-81731-A-10.wav
    2-82367-A-10.wav
    2-87781-A-10.wav
    3-132852-A-10.wav
    3-140774-A-10.wav
    3-142005-A-10.wav
    3-142006-A-10.wav
    3-143929-A-10.wav
    3-157149-A-10.wav
    3-157487-A-10.wav
    3-157615-A-10.wav
    4-160999-A-10.wav
    4-161127-A-10.wav
    4-163264-A-10.wav
    4-164206-A-10.wav
    4-166661-A-10.wav
    4-177250-A-10.wav
    4-180380-A-10.wav
    4-181286-A-10.wav
    5-181766-A-10.wav
    5-188655-A-10.wav
    5-193339-A-10.wav
    5-194892-A-10.wav
    5-195710-A-10.wav
    5-198321-A-10.wav
    5-202898-A-10.wav
    5-203739-A-10.wav
    1-26806-A-1.wav
    1-27724-A-1.wav
    1-34119-A-1.wav
    1-34119-B-1.wav
    1-39923-A-1.wav
    1-40730-A-1.wav
    1-43382-A-1.wav
    1-44831-A-1.wav
    2-100786-A-1.wav
    2-65750-A-1.wav
    2-71162-A-1.wav
    2-81270-A-1.wav
    2-95035-A-1.wav
    2-95258-A-1.wav
    2-95258-B-1.wav
    2-96460-A-1.wav
    3-107219-A-1.wav
    3-116135-A-1.wav
    3-134049-A-1.wav
    3-137152-A-1.wav
    3-145382-A-1.wav
    3-149189-A-1.wav
    3-154957-A-1.wav
    3-163288-A-1.wav
    4-164021-A-1.wav
    4-164064-A-1.wav
    4-164064-B-1.wav
    4-164064-C-1.wav
    4-164859-A-1.wav
    4-170078-A-1.wav
    4-183487-A-1.wav
    4-208021-A-1.wav
    5-194930-A-1.wav
    5-194930-B-1.wav
    5-200334-A-1.wav
    5-200334-B-1.wav
    5-200339-A-1.wav
    5-233160-A-1.wav
    5-234879-A-1.wav
    5-234879-B-1.wav
    1-28135-A-11.wav
    1-28135-B-11.wav
    1-39901-A-11.wav
    1-39901-B-11.wav
    1-43760-A-11.wav
    1-61252-A-11.wav
    1-91359-A-11.wav
    1-91359-B-11.wav
    2-102852-A-11.wav
    2-124662-A-11.wav
    2-125966-A-11.wav
    2-132157-A-11.wav
    2-132157-B-11.wav
    2-133863-A-11.wav
    2-137162-A-11.wav
    2-155801-A-11.wav
    3-144827-A-11.wav
    3-144827-B-11.wav
    3-155642-A-11.wav
    3-155642-B-11.wav
    3-164120-A-11.wav
    3-164630-A-11.wav
    3-166422-A-11.wav
    3-187710-A-11.wav
    4-167063-A-11.wav
    4-167063-B-11.wav
    4-167063-C-11.wav
    4-182613-A-11.wav
    4-182613-B-11.wav
    4-195497-A-11.wav
    4-195497-B-11.wav
    4-204618-A-11.wav
    5-200461-A-11.wav
    5-200461-B-11.wav
    5-208810-A-11.wav
    5-208810-B-11.wav
    5-213077-A-11.wav
    5-219379-A-11.wav
    5-219379-B-11.wav
    5-219379-C-11.wav
    1-26143-A-21.wav
    1-29680-A-21.wav
    1-31748-A-21.wav
    1-47273-A-21.wav
    1-47274-A-21.wav
    1-54505-A-21.wav
    1-59324-A-21.wav
    1-81883-A-21.wav
    2-109505-A-21.wav
    2-118104-A-21.wav
    2-119102-A-21.wav
    2-128631-A-21.wav
    2-130978-A-21.wav
    2-130979-A-21.wav
    2-82538-A-21.wav
    2-93030-A-21.wav
    3-141684-A-21.wav
    3-142601-A-21.wav
    3-142605-A-21.wav
    3-143119-A-21.wav
    3-144692-A-21.wav
    3-148330-A-21.wav
    3-150231-A-21.wav
    3-156558-A-21.wav
    4-156843-A-21.wav
    4-156844-A-21.wav
    4-157297-A-21.wav
    4-167642-A-21.wav
    4-171519-A-21.wav
    4-184434-A-21.wav
    4-185415-A-21.wav
    4-185619-A-21.wav
    5-187979-A-21.wav
    5-194533-A-21.wav
    5-201274-A-21.wav
    5-202220-A-21.wav
    5-220026-A-21.wav
    5-220027-A-21.wav
    5-221518-A-21.wav
    5-221593-A-21.wav
    """

esc10_class = """
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    chainsaw
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    clock_tick
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crackling_fire
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    crying_baby
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    dog
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    helicopter
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rain
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    rooster
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sea_waves
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    sneezing
    """
# %%
def _wav2array(nchannels, sampwidth, data):
    """data must be the string containing the bytes from the wav file."""
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
        raw_bytes = np.fromstring(data, dtype=np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = np.fromstring(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(-1, nchannels)
    return result


def readwav(file):
    """
    Read a wav file.
    Returns the frame rate, sample width (in bytes) and a numpy array
    containing the data.
    This function does not read compressed wav files.
    """
    wav = wave.open(file)
    rate = wav.getframerate()
    nchannels = wav.getnchannels()
    sampwidth = wav.getsampwidth()
    nframes = wav.getnframes()
    data = wav.readframes(nframes)
    wav.close()
    array = _wav2array(nchannels, sampwidth, data)
    return rate, array


def spectro(x, bw=199, overlap=0.2, fs=44.1e3, showplot=False):
    """
    this will calc the spectrogram, using the setting of YOU's paper
    :param x:
    :param bw:
    :param overlap:
    :param fs:
    :param showplot:
    :return:
    """
    f, t, sx = sg.spectrogram(x,window=sg.windows.hann(bw), fs=fs, nfft=int(bw*2), noverlap=int(bw*overlap))
    sp = np.log(np.abs(sx) ** 2 + 5e-32)
    if showplot:
        plt.figure()
        plt.title('Spectrogram')
        plt.imshow(sx, aspect='auto', interpolation='None')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
    return sp, sx


def downsample(x, t_len=200, f_len=200):
    """
    This function will downsample the audio signal in frequency domain
    by converting to spectrogram and doing the downsampling on the image
    :param x: input data N by T
    :return:
    """
    newx = np.zeros((f_len, t_len))
    _, sx = spectro(x)
    newx = cv2.resize(sx, (t_len, f_len), interpolation=cv2.INTER_CUBIC)
    return newx


def draw_sig(pool, must=False):
    """draw a signal from the pool

    Args:
        pool ([type]): [description]
        must (bool, optional): [description]. Defaults to False.
    """
    X, Y = pool
    n_class = Y.shape[1]
    n_per_class = Y.shape[0]//n_class# perclass samples
    if must:
        class_idx = list(range(10))  # 10 classes in ESC10
    else:
        class_idx = list(range(11))  # 1 extra for null
    sig_idx = list(range(n_per_class))  
    np.random.shuffle(sig_idx)
    np.random.shuffle(class_idx)
    idx = class_idx[0]*n_per_class + sig_idx[0]
    
    if class_idx[0] == 10:
        x, y = X[0], Y[0]
        return x-x, y-y  # return 0s
    else:
        x, y = X[idx], Y[idx]
        return x, y

def pack_data_label(pool, n, f_len, t_len):
    """the data are generated in the following way:
    suppose there are 5 slots, for each slot there could one non-mixture sample or not
    all the 5 slots must contain at least one non-mixture sample

    for the Not concatanated
    5 slots are stored in a list of data

    for concatanated data in a bag
    5 slots are stored as a tensor
    """
    #"the bags of data Not concatanated"
    "choose 1 of the 5 slots must contain single"
    slots = list(range(5))
    X_bag = np.random.rand(n, 5, f_len, t_len)  
    Y = np.random.rand(n, 10)
    y_detail = np.random.rand(n, 5, 10)
    for ii in range(n):
        np.random.shuffle(slots)
        y = np.random.rand(5, 10) # 5 slots, 10 classes
        for i in range(5):    
            i_slot = slots[i] 
            if i == 0 : 
                X_bag[ii, i_slot], y[i] = draw_sig(pool, must=True) # must contain signal
            else:
                X_bag[ii, i_slot], y[i] = draw_sig(pool)
            y_detail[ii, i_slot] = y[i]
        y = y.sum(0)
        y[y>0] = 1.0
        Y[ii] = y

    #"concatanated data in a bag"
    X = np.moveaxis(X_bag, 1, 2).reshape(X_bag.shape[0], f_len, 5*t_len)
    return X_bag, X, Y, y_detail


def split_data(data, label, n_test=50):
    """split data from trian and test

    Args:
        data ([type]): [tensor, n_sample*n_f*n_t]
        label ([type]): [matrix, n_sample*n_classes]
        portion (int, optional): [number of test samples]. Defaults to 50.
    """
    n_class = label.shape[1]  # 10 classes
    n_test_class = n_test//n_class  # 5
    n_samp_per_class =  data.shape[0]//n_class # 40
    nn = n_samp_per_class - n_test_class  #35
    x_tr = np.random.rand(nn*n_class, data.shape[1], data.shape[2])
    y_tr = np.random.rand(nn*n_class, label.shape[1])
    x_te = np.random.rand(n_test_class*n_class, data.shape[1], data.shape[2])
    y_te = np.random.rand(n_test_class*n_class, label.shape[1])
    for i in range(n_class):
        x_tr[i*nn:i*nn+nn] = data[i*n_samp_per_class:i*n_samp_per_class+nn]
        y_tr[i*nn:i*nn+nn] = label[i*n_samp_per_class:i*n_samp_per_class+nn]
        x_te[i*n_test_class:(i+1)*n_test_class] = data[i*n_samp_per_class+nn:(i+1)*n_samp_per_class]
        y_te[i*n_test_class:(i+1)*n_test_class] = label[i*n_samp_per_class+nn:(i+1)*n_samp_per_class]
    return x_tr, y_tr , x_te, y_te