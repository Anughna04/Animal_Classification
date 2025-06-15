# 🐾 Animal Image Classification

This project leverages **Transfer Learning** with the **MobileNet** architecture to classify animals from images. MobileNet is a lightweight and efficient deep learning model, making it ideal for real-time and mobile applications. The model is fine-tuned on a custom animal dataset to recognize multiple species with high accuracy.

---

## 📌 Features

- 🔍 **Image Preprocessing**: Resizing, normalization, and data augmentation
- 🧠 **Transfer learning**: Pretrained MobileNetV3 for classificcation
- 📊 **Model Evaluation**: Accuracy, loss, and confusion matrix visualization
- 🗂️ **Train/Test Split**: Organized pipeline for splitting datasets
- 🚀 **Model Training**: Easily configurable epochs, batch size, and optimizer
- 🧪 **Inference**: Classify new animal images with a single command

---

## 🛠️ Tools & Technologies Used

| Category            | Tools/Technologies                      |
|---------------------|------------------------------------------|
| Programming Language| Python 3.10                              |
| Deep Learning       | TensorFlow, Keras                        |
| Data Handling       | NumPy, Pandas                            |
| Image Processing    | OpenCV, Matplotlib, PIL                  |                      
| Version Control     | Git & GitHub                             |

---

## 📁 Project Structure

    Animal_Classification/
    │
    ├── train_model.py # Model training script
    ├── predict.py # Predict new images
    ├── data/ # Contains train/test images
    │ ├── train/
    │ └── test/
    ├── model/ # Saved model and weights
    │ ├── model.json
    │ └── final_weights.h5
    ├── utils/ # Utility functions (e.g., for loading data)
    ├── requirements.txt # Dependencies
    └── README.md # Project documentation


## ▶️ How to Run the Project

### 1. 📦 Install Dependencies

Make sure you have Python 3.10+ installed. Then run:

```bash
pip install -r requirements.txt
```

2. 🏗 Prepare Dataset
Ensure your dataset is structured like this:

        data/
        ├── train/
        │   ├── cat/
        │   ├── dog/
        │   └── elephant/
        └── test/
            ├── cat/
            ├── dog/
            └── elephant/
You can add any number of animal classes.

3. 🧠 Train the Model

       python train_model.py
   
4.Run on streamlit

       streamlit run app.py
       
