#!/usr/bin/env python
# coding: utf-8

# In[27]:


import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display

sample_rate, samples = wavfile.read('/Users/oreoluwa/Downloads/20220307_080000.WAV')
print(samples.shape,sample_rate)
plt.plot(samples[0:10000])
plt.show()


# In[28]:


import cv2
import numpy as np
from skimage import morphology
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage import color, morphology
from skimage import io


# In[29]:


path = '/Users/oreoluwa/Downloads/20220307_080000.WAV'
scale, sr = librosa.load(path)
X = librosa.stft(scale)
Xdb = librosa.amplitude_to_db(abs(X))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')


# In[30]:


SOUND_DIR = '/Users/oreoluwa/Downloads/20220307_080000.WAV'
            

# load the mp3 file
signal, sr = librosa.load(SOUND_DIR, duration=10)  # sr = sampling rate

# plot recording signal
plt.figure(figsize=(10, 4))
librosa.display.waveshow(signal, sr=sr)
plt.title("Monophonic/Waveform")
plt.show()


# In[31]:


import librosa
import librosa.display
from matplotlib import pyplot as plt
import numpy as np


# In[32]:


# Plot spectogram
plt.figure(figsize=(10, 4))
D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
librosa.display.specshow(D, y_axis="linear")
plt.colorbar(format="%+2.0f dB")
plt.title("Linear-frequency power spectrogram")
plt.show()

# Plot mel-spectrogram
N_FFT = 1024
HOP_SIZE = 1024
N_MELS = 128
WIN_SIZE = 1024
WINDOW_TYPE = "hann"
FEATURE = "mel"
FMIN = 0

S = librosa.feature.melspectrogram(
    y=signal,
    sr=sr,
    n_fft=N_FFT,
    hop_length=HOP_SIZE,
    n_mels=N_MELS,
    htk=True,
    fmin=FMIN,
    fmax=sr / 2,
)

plt.figure(figsize=(10, 4))
librosa.display.specshow(
    librosa.power_to_db(S ** 2, ref=np.max), fmin=FMIN, y_axis="linear"
)
plt.colorbar(format="%+2.0f dB")
plt.title("Mel-scaled spectrogram")
plt.show()

# Plot mel-spectrogram with high-pass filter
N_FFT = 1024
HOP_SIZE = 1024
N_MELS = 128
WIN_SIZE = 1024
WINDOW_TYPE = "hann"
FEATURE = "mel"
FMIN = 1400

S = librosa.feature.melspectrogram(
    y=signal,
    sr=sr,
    n_fft=N_FFT,
    hop_length=HOP_SIZE,
    n_mels=N_MELS,
    htk=True,
    fmin=FMIN,
    fmax=sr / 2,
)

plt.figure(figsize=(10, 4))
librosa.display.specshow(
    librosa.power_to_db(S ** 2, ref=np.max), fmin=FMIN, y_axis="linear"
)
plt.colorbar(format="%+2.0f dB")
plt.title("Mel-scaled spectrogram with high-pass filter - 10 seconds")
plt.show()


# In[33]:


Output=librosa.power_to_db(S ** 2, ref=np.max)
plt.imshow(Output)
plt.show()


# In[34]:


S.shape


# In[35]:


cleaned.shape


# In[36]:


Output.max()


# In[37]:


plt.plot(Output.mean(axis=1))


# In[38]:


Average = Output.mean(axis=1)
Average = Average.reshape(Average.shape+(1,))
plt.imshow(Output-Average)
plt.show()


# In[39]:


plt.imshow((Output-Average)>5)
plt.show()
(Output-Average).max()
img = SOUND_DIR


# In[40]:


img = (Output-Average)>7
##img = (img*255).astype(np.uint8)

kernel = np.ones((5, 5), np.uint8)
##opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cleaned = morphology.remove_small_objects(img, min_size=15, connectivity=2)
plt.imshow(cleaned)


# In[41]:


def detect_birdsound():
 img = (Output-Average)>17
##img = (img*255).astype(np.uint8)

kernel = np.ones((5, 5), np.uint8)
##opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cleaned = morphology.remove_small_objects(img, min_size=50, connectivity=2)
plt.imshow(cleaned)
print(detect_birdsound)


# In[42]:


# Python program to explain os.mkdir() method
  
# importing os module
import os
  
# Directory
directory = "SOUNDS"
  
# Parent Directory path
parent_dir = "D:/Users/oreoluwa/Desktop/" , 
  
# Path
path = os.path.join("D:/Users/oreoluwa/Desktop/", "SOUNDS")
  
# Create the directory
# 'SOUNDS' in
# '/home / User / Documents'
os.mkdir("/Users/oreoluwa/SOUNDS")
print("Directory '% s' created" % directory)
  
# Directory
directory = "/Users/oreoluwa/Downloads/SOUNDS"
  
# Parent Directory path
parent_dir = "D:/Users/oreoluwa/Desktop/PROJECTS"

# mode
mode = 0o666
  
# Path
path = os.path.join(parent_dir, directory)
  
# Create the directory
# 'SOUNDS' in
# '/home / User / Documents'
# with mode 0o666
os.mkdir(path, mode)
print("Directory '% s' created" % directory)

  


# In[43]:


import os


# In[44]:


def waveform_to_sonogram(waveform,sampling):
    return to_db(to_sonogram(waveform))

def find_songs(waveform):
    img= waveform_to_sonogram(waveform,sampling)
    img= img - img.mean(axis = 1).reshape(())
    img= (img>calc) >20
    
    return morphology.remove_small_objects(img,6,100)

for filename in os.listdir("SOUNDS"):
    waveform = read_waveform(Output)
    detected = find_SONGS(waveform)
    plt.imshow(detected)
    plt.savefig()
    plt.close()
    


# In[45]:


def waveform_to_sonogram(waveform,sampling):
    return to_db(to_sonogram(waveform))

def find_songs(waveform):
    img= waveform_to_sonogram(waveform,sampling)
    img= img - img.mean(axis = 1).reshape(())
    img= (img>calc) >20
    
    return morphology.remove_small_objects(img,6,100)

for file in os.listdir("SOUNDS"):
    waveform = read_waveform('/Users/oreoluwa/Downloads/20220307_080000.WAV')
    detected = find_SONGS(waveform)
    plt.imshow(detected)
    plt.savefig()
    plt.close()
   


# In[46]:


def waveform_to_sonogram(waveform,sampling):
    return to_db(to_sonogram(waveform))

def find_songs(waveform):
    img= waveform_to_sonogram(waveform,sampling)
    img= img - img.mean(axis = 1).reshape(())
    img= (img>calc) >17
    
    return morphology.remove_small_objects(img,6,100)

for file in os.listdir("SOUNDS"):
    waveform = read_waveform('/Users/oreoluwa/Downloads/20220307_080000.WAV')
    detected = find_SONGS(waveform)
    plt.imshow(detected)
    plt.savefig()
    plt.close()
cleaned = morphology.remove_small_objects(img, min_size=50, connectivity=2)
plt.imshow(cleaned)


# In[47]:


def waveform_to_sonogram(waveform,sampling):
    return to_db(to_sonogram(waveform))
    

def find_songs(waveform):
    img= waveform_to_sonogram(waveform,sampling)
    img= img - img.mean(axis = 1).reshape(())
    img= (img>calc) >17
    
    return morphology.remove_small_objects(img,6,100)

for file in os.listdir("SOUNDS"):
    waveform = read_waveform('/Users/oreoluwa/Downloads/20220307_080000.WAV')
    detected = find_SONGS(waveform)
    plt.imshow(detected)
    plt.savefig()
    plt.close()
cleaned = morphology.remove_small_objects(img, min_size=10, connectivity=2)
plt.imshow(cleaned)

