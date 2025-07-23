
import os
import shutil
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# STEP 1: Organize Dataset
def prepare_dataset():
    source_normal = 'D:/Guvi_Project5/Dataset/Normal Chest X-rays'
    source_tb = 'D:/Guvi_Project5/Dataset/TB Chest X-rays'
    target_base = 'D:/Guvi_Project5/Dataset_Prepared'

    os.makedirs(os.path.join(target_base, 'Normal'), exist_ok=True)
    os.makedirs(os.path.join(target_base, 'TB'), exist_ok=True)

    for src_dir, dst_dir in [(source_normal, 'Normal'), (source_tb, 'TB')]:
        for filename in os.listdir(src_dir):
            src = os.path.join(src_dir, filename)
            dst = os.path.join(target_base, dst_dir, filename)
            shutil.copyfile(src, dst)
    
    print("✅ Dataset copied to:", target_base)

# STEP 2: Train Model
def train_model():
    data_dir = 'D:/Guvi_Project5/Dataset_Prepared'
    model_path = 'D:/Guvi_Project5/models/best_model.h5'
    os.makedirs('D:/Guvi_Project5/models', exist_ok=True)

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(
        data_dir, target_size=(224, 224), batch_size=32,
        class_mode='binary', subset='training'
    )

    val_gen = train_datagen.flow_from_directory(
        data_dir, target_size=(224, 224), batch_size=32,
        class_mode='binary', subset='validation'
    )

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_gen, validation_data=val_gen, epochs=5)
    model.save(model_path)
    print("✅ Model trained and saved to:", model_path)

# STEP 3: Evaluate Model
def evaluate_model():
    model_path = 'D:/Guvi_Project5/models/best_model.h5'
    model = tf.keras.models.load_model(model_path)

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        'D:/Guvi_Project5/Dataset_Prepared',
        target_size=(224, 224), batch_size=32,
        class_mode='binary', shuffle=False
    )

    predictions = model.predict(test_gen)
    predicted_classes = np.where(predictions > 0.5, 1, 0)
    true_classes = test_gen.classes
    labels = list(test_gen.class_indices.keys())

    print("✅ Evaluation Metrics:")
    print(classification_report(true_classes, predicted_classes, target_names=labels))

# STEP 4: Run Streamlit App

# -------------------------------
# STREAMLIT PAGE 1: Introduction
# -------------------------------
def show_home():
    st.title("🩺 Tuberculosis Detection Using Deep Learning")
    st.image("/home/ubuntu/TB_Project/Dataset/TB_Project_Image.png", use_container_width=True)
    st.markdown("""
        This project uses a Convolutional Neural Network with **Transfer Learning (ResNet50)** to classify chest X-rays as:
        - **Normal**
        - **Tuberculosis (TB)**

        ### 👇 Features:
        - Deep Learning with TensorFlow
        - Streamlit Web Interface
        - Image Upload & Prediction
        - AWS Deployment Ready

        ---
        """)
    

# -------------------------------
# STREAMLIT PAGE 2: TB Detection
# -------------------------------
def show_detection():
    st.title("🧪 TB Detection from Chest X-ray")
    model_path = '/home/ubuntu/TB_Project/models/best_model.h5'
    model = tf.keras.models.load_model(model_path)

    uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0][0]
        st.write(f"🧠 Model Confidence: `{pred:.2f}`")

        if pred > 0.5:
            st.error("❌ Prediction: Tuberculosis Detected")
        else:
            st.success("✅ Prediction: Normal")

# -------------------------------
# STREAMLIT PAGE 3: About Me
# -------------------------------
def show_about():
    st.title("👤 About Me")
    st.markdown("""
    **👨‍💻 Developer:** *Keerthana R R*  
    **📧 Email:** keerthanarr1997@gmail.com  
    **🛠 Tech Stack:**  
    - Python, TensorFlow, Keras  
    - Streamlit, OpenCV, NumPy  
    - AWS EC2 for Deployment

    [![GitHub](https://img.shields.io/badge/GitHub-View%20Profile-black)](https://github.com/KeerthanaRR97)
    """)

# -------------------------------
# MAIN STREAMLIT APP NAVIGATION
# -------------------------------
def run_app():
    st.set_page_config(page_title="TB Detection App", layout="centered")

    menu = st.sidebar.radio("📌 Navigate", ["🏠 Home", "🧪 TB Detection", "👤 About Me"])

    if menu == "🏠 Home":
        show_home()
    elif menu == "🧪 TB Detection":
        show_detection()
    elif menu == "👤 About Me":
        show_about()

# MAIN CONTROLLER
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TB Detection Project")
    parser.add_argument('--prepare', action='store_true', help="Step 1: Prepare dataset")
    parser.add_argument('--train', action='store_true', help="Step 2: Train model")
    parser.add_argument('--evaluate', action='store_true', help="Step 3: Evaluate model")
    parser.add_argument('--app', action='store_true', help="Step 4: Launch Streamlit app")

    args = parser.parse_args()

    if args.prepare:
        prepare_dataset()
    elif args.train:
        train_model()
    elif args.evaluate:
        evaluate_model()
    elif args.app:
        run_app()
    else:
        print("📌 Usage:\n"
              "  python main.py --prepare   # Prepare dataset\n"
              "  python main.py --train     # Train model\n"
              "  python main.py --evaluate  # Evaluate model\n"
              "  streamlit run main.py -- --app  # Run Streamlit app")
