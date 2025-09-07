# Urban Sound Classification with Deep Learning 🎵🤖

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Classify urban sounds using a **Deep Learning model** trained on the **UrbanSound8K dataset**.  
The model uses a **Multi-Layer Perceptron (MLP)** and features extracted with **MFCCs** from audio files.


# Clone this repository
git clone [https://github.com/ommishra03/urban-sound-classification.git](https://github.com/ommishra03/Urban-Sound-Classification-with-Deep-Learning)
cd urban-sound-classification

# Install dependencies
pip install numpy pandas librosa scikit-learn tensorflow keras

# Launch Jupyter Notebook
jupyter notebook Urban_Sound_Classification.ipynb
````

## 📂 Dataset

The project uses the **UrbanSound8K dataset**:
[Download from Kaggle](https://www.kaggle.com/urbansound8k/urbansound8k)

**Directory structure expected:**

```
UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── fold2/
│   └── ...
└── metadata/
    └── UrbanSound8K.csv
```

---

## 🛠️ Features & Workflow

1. **Data Loading**

```python
import librosa
import pandas as pd

metadata = pd.read_csv('metadata/UrbanSound8K.csv')
y, sr = librosa.load('audio/fold1/101415-3-0-2.wav')
```

2. **Feature Extraction (MFCC)**

```python
import numpy as np

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
mfccs_processed = np.mean(mfccs.T, axis=0)
```

3. **Model Training**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(256, activation='relu', input_shape=(40,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
```

4. **Evaluation**

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
```

---

## 📊 Results

* High classification accuracy for urban sounds
* Confusion matrix and training/validation loss plots show convergence
* Example classes: `dog_bark`, `car_horn`, `street_music`, etc.

---

## 🛠️ Built With

* **Python** – Core programming language
* **Keras/TensorFlow** – Deep learning
* **librosa** – Audio analysis & feature extraction
* **pandas & numpy** – Data handling
* **matplotlib & seaborn** – Visualization

---

## 👤 Author

**Om Mishra**: [GitHub](https://github.com/ommishra03), [Linkedin](linkedin.com/in/om-mishra-a62991289), [E-mail](mailto:ommishra1729@gmail.com)

---

## 🙏 Acknowledgments

* [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html) creators
* [Kaggle](https://www.kaggle.com/) for notebook environment & resources

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```

Do you want me to do that next?
```
