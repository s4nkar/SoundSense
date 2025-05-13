from torch import nn
from transformers import Wav2Vec2Model
import torch.nn.functional as F
import numpy as np
import librosa

# Wav2Vec2 Model definition
class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_labels):
        super(EmotionRecognitionModel, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Sequential(
            nn.Linear(self.wav2vec2.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        pooled_output = hidden_states.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits

# CNN Model with Global Average Pooling
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.85)
        self.fc2 = nn.Linear(256, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.training:
            x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# Custom Audio Augmentations
def add_gaussian_noise(audio, amplitude=0.015):
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, amplitude, len(audio))
        return audio + noise
    return audio

def pitch_shift(audio, sr, n_steps):
    if np.random.rand() < 0.5:
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    return audio

def time_stretch(audio, rate):
    if np.random.rand() < 0.5:
        return librosa.effects.time_stretch(audio, rate=rate)
    return audio

def apply_augmentations(audio, sr):
    audio = add_gaussian_noise(audio, amplitude=np.random.uniform(0.005, 0.015))
    audio = pitch_shift(audio, sr, n_steps=np.random.uniform(-4, 4))
    audio = time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
    return audio

# Feature Extraction with Normalization
def extract_features(file_path, augment=True):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        if len(y) < 0.5 * sr:  # Skip clips <0.5s
            print(f"Skipping {file_path}: Too short")
            return None
        if augment:
            y = apply_augmentations(y, sr)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel = mel_spectrogram_db.T
        if mel.shape[0] > 128:
            mel = mel[:128, :]
        elif mel.shape[0] < 128:
            mel = np.pad(mel, ((0, 128 - mel.shape[0]), (0, 0)), mode='constant')
        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        return mel
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
def ml_model_extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0) #MFCC (Mel-frequency cepstral coefficients) – captures timbral texture.
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0) #Chroma – reflects pitch class energy (musical notes).
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0) #Spectral Contrast – helps with timbre and brightness.
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0) #ZCR (Zero Crossing Rate) – good for detecting noisiness or fricatives.
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0) #RMS (Root Mean Square Energy) – correlates with loudness.

    return np.hstack([mfcc, chroma, contrast, zcr, rms])