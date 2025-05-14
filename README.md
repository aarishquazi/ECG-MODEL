# 🫀 12-Lead ECG Classification with Deep Learning

A deep learning-based image classification project using **12-lead ECG images**. Trained on a labeled dataset of cardiac conditions, this model leverages **MobileNetV2** to classify ECGs with **90%+ accuracy**.

## 🚀 Overview

This project uses transfer learning with MobileNetV2 to classify ECG images into multiple diagnostic categories. It is designed to be lightweight and accurate, suitable for deployment in real-time health monitoring dashboards or embedded systems.

## 🧠 Model Highlights

- 📦 **Architecture**: MobileNetV2 (pretrained on ImageNet)
- 🔍 **Custom Layers**:
  - `GlobalAveragePooling2D`
  - `Dropout(0.2)`
  - `Dense` layer with Softmax activation
- 🧪 **Loss**: Categorical Crossentropy
- 🧮 **Optimizer**: Adam
- 📊 **Accuracy Achieved**: **90%+**

## 📁 Project Structure

```
📦 ecg-classifier/
├── ECG_PROJECT.ipynb     # Jupyter notebook with model pipeline
├── model.h5              # Trained Keras model
├── requirements.txt      # Python dependencies
├── /images               # ECG dataset (organized by class)
└── README.md             # Project documentation
```

## 🔧 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ecg-classifier.git
   cd ecg-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook ECG_PROJECT.ipynb
   ```

## 🧪 Sample Training Code

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input

base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
output = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)
```

## 📈 Results

| Metric | Value |
|--------|-------|
| Accuracy | **90%+** |
| Epochs | 20 |
| Model Size | ~15MB |

⚠️ Your results may vary slightly depending on dataset size, quality, and preprocessing steps.

## 🔮 Future Enhancements

* ✅ Add Grad-CAM visualization for interpretability
* 📉 Implement early stopping & learning rate scheduling
* 💾 Export results as medical PDF reports
* 📱 Deploy model to a web/mobile health app

## 📋 Requirements

```
tensorflow
numpy
pandas
scipy
matplotlib
opencv-python
Pillow
scikit-learn
imgaug
jupyter
notebook
neurokit2
```

Install with:
```bash
pip install -r requirements.txt
```

## 🧑‍💻 Author

Aarish Quazi 
Python & AI Developer  
📍 Jaipur
aarishquazi@gmail.com

## 📝 License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute.

## ❤️ Contributions

Have suggestions or improvements? Feel free to open an issue or submit a PR!
