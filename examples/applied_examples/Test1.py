import numpy as np
import pandas as pd
import mne
from braindecode import EEGClassifier
from skorch.dataset import ValidSplit
from braindecode.models import ShallowFBCSPNet

# Load your own CSV data
# Assume the CSV file has columns representing EEG channels (e.g., 'C3', 'C4', 'Cz')
data = pd.read_csv()

# Separate features (X) and labels (y)
# Assume the first columns are features and the last column is the labels
X = data.iloc[:, :-1].values  # all columns except the last one
y = data.iloc[:, -1].values   # the last column as the labels

# Create MNE Epochs object
info = mne.create_info(ch_names=["EEG.AF3", "EEG.F7", "EEG.F3", "EEG.FC5",
                                 "EEG.T7", "EEG.P7", "EEG.O1", "EEG.O2",
                                 "EEG.P8", "EEG.T8", "EEG.FC6", "EEG.F4",
                                 "EEG.F8", "EEG.AF4"], sfreq=128.0, ch_types="eeg")
epochs = mne.EpochsArray(X, info=info)

# Now we have the data in the correct format: epochs (mne.EpochsArray), and labels y

# Initialize your model
net = EEGClassifier(
    "ShallowFBCSPNet",
    module__final_conv_length="auto",
    train_split=ValidSplit(0.2),
    # You can pass additional parameters here if necessary
)

# Fit the model
net.fit(epochs, y)

# The trained model is now available in net.module_
print(net.module_)
