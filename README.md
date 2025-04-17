This project focuses on detecting bird sounds from an audio recording (WAV file) using audio signal processing and spectrogram analysis. You used tools like Librosa, OpenCV, and scikit-image's morphology functions to:

Read and analyze sound files

Visualize waveforms and spectrograms

Create mel-scaled spectrograms and high-pass filter variants

Identify potential bird sound patterns

Apply image-processing techniques to clean and highlight sound events

Set up a structure to batch-process multiple audio files for bird detection

📂 Step-by-Step Breakdown
1. 📥 Load the Audio File
python
Copy
Edit
sample_rate, samples = wavfile.read('20220307_080000.WAV')
I read a .WAV file and plotted a small segment of it to view the waveform.

samples.shape = (14400000,) means it's a 1D mono audio with 14.4M data points sampled at 48kHz (i.e., very high resolution).

2. 📊 Basic Signal Visualization (Waveform & STFT Spectrogram)
python
Copy
Edit
scale, sr = librosa.load(path)
X = librosa.stft(scale)
I computed the Short-Time Fourier Transform (STFT) to visualize frequency components over time.

librosa.display.specshow() is used to generate a time-frequency spectrogram in decibels.

3. 🔉 10-Second Segment & Waveform Plot
python
Copy
Edit
signal, sr = librosa.load(SOUND_DIR, duration=10)
You limited your analysis to the first 10 seconds of audio for efficiency.

Then plotted the waveform to understand amplitude over time.

4. 🔬 Spectrograms for Frequency Analysis
I plotted two spectrograms:

Linear-frequency spectrogram using librosa.stft

Mel-scaled spectrogram, more aligned with human auditory perception

I experimented with fmin (minimum frequency) to apply a high-pass filter, excluding lower-frequency noise (like background hum) to isolate bird sounds, which are often higher-pitched.

5. 📈 Spectrogram Post-processing (Enhancement & Subtraction)
python
Copy
Edit
Output = librosa.power_to_db(S ** 2, ref=np.max)
Average = Output.mean(axis=1).reshape((-1, 1))
plt.imshow(Output - Average)
You normalized the spectrogram by subtracting the frequency band’s average.

This highlights time-frequency areas with strong deviation, which may contain birdsong.

6. 🧹 Binary Masking and Morphological Cleaning
python
Copy
Edit
img = (Output - Average) > 7
cleaned = morphology.remove_small_objects(img, min_size=50, connectivity=2)
Converted the spectrogram into a binary mask (True/False) where potential bird sounds occur.

Cleaned up noise using remove_small_objects() to eliminate tiny, irrelevant detections (non-bird audio).

7. 🐦 Bird Sound Detection Function
python
Copy
Edit
def detect_birdsound():
    img = (Output - Average) > 17
    cleaned = morphology.remove_small_objects(img, min_size=50, connectivity=2)
    plt.imshow(cleaned)
This function encapsulates the detection logic.

A higher threshold (>17) is used to highlight only the loudest/highest energy bursts, likely birdsong.

8. 📁 Directory & File Setup
python
Copy
Edit
os.mkdir("/Users/oreoluwa/SOUNDS")
I created a folder to store sound files or output detections.

Encountered FileExistsError, which is expected if the folder already exists (you can fix with if not os.path.exists(path):)

