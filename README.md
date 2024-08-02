# Gender-identification-from-voice
This code implements a system for gender identification from voice recordings using Gaussian Mixture Models (GMMs). It includes a FeaturesExtractor class to extract Mel Frequency Cepstral Coefficients (MFCC) along with their deltas and double deltas from audio files. The ModelsTrainer class collects these features from training data (separate male and female voice recordings) to train GMMs for each gender. The trained models are saved using pickle. The GenderIdentifier class loads these models to classify test audio files, determining the gender based on the log-likelihood scores from the GMMs. The accuracy of the identification process is then calculated and printed.
