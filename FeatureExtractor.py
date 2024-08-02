import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
from python_speech_features import mfcc, delta

class FeaturesExtractor:
    def __init__(self):
        pass

    def extract_features(self, audio_path):
        """
        Extract voice features including the Mel Frequency Cepstral Coefficient (MFCC)
        from an audio using the python_speech_features module, performs Cepstral Mean
        Normalization (CMS) and combine it with MFCC deltas and the MFCC double deltas.

        Args:
            audio_path (str) : path to wave file without silent moments.

        Returns:
            (array) : Extracted features matrix.
        """
        rate, audio = read(audio_path)
        mfcc_feature = mfcc(
            audio,
            rate,
            winlen=0.05,
            winstep=0.01,
            numcep=5,
            nfilt=30,
            nfft=512,
            appendEnergy=True
        )
        mfcc_feature = preprocessing.scale(mfcc_feature)
        deltas = delta(mfcc_feature, 2)
        double_deltas = delta(deltas, 2)
        combined = np.hstack((mfcc_feature, deltas, double_deltas))
        return combined
