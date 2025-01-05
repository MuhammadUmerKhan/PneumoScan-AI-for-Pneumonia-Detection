import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Streamlit page configuration
st.set_page_config(
    page_title="Pneumonia Disease Detector",
    page_icon="🤒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            margin-top: -10px;
        }
        /* Main Title */
        .main-title {
            font-size: 2.5em;
            font-weight: bold;
            color: #C0C0C0;
            text-align: center;
            margin-bottom: 20px;
        }
        /* Section Titles */
        .section-title {
            font-size: 1.8em;
            color: #C0C0C0;
            font-weight: bold;
            margin-top: 30px;
            text-align: left;
        }
        /* Tab Title Customization */
        .stTab {
            font-size: 1.4em;  /* Increase tab title font size */
            font-weight: bold;
            color: #2980B9;
        }
        /* Section Content */
        .section-content{
            text-align: center;
        }
        /* Home Page Content */
        .intro-title {
            font-size: 2.5em;
            color: #00ce39;
            font-weight: bold;
            text-align: center;
        }
        .intro-subtitle {
            font-size: 1.2em;
            color: #017721;
            text-align: center;
        }
        .content {
            font-size: 1em;
            color: #7F8C8D;
            text-align: justify;
            line-height: 1.6;
        }
        .highlight {
            # color: #068327;
            font-weight: bold;
        }
        /* Separator Line */
        .separator {
            height: 2px;
            background-color: #BDC3C7;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        /* Prediction Text Styling */
        .prediction-text {
            font-size: 2em;
            font-weight: bold;
            color: #2980B9;
            text-align: center;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        }
        /* Footer */
        .footer {
            font-size: 14px;
            color: #95A5A6;
            margin-top: 20px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


# Title Heading (appears above tabs and remains on all pages)
st.markdown('<div class="main-title">🩻 Welcome to the Pneumonia Disease Classification Tool 🌱</div>', unsafe_allow_html=True)

# Tab layout
tab1, tab2 = st.tabs(["🏠 Dashboard", "🩻 Test X-ray Report"])

# First Tab: Home
with tab1:
    st.markdown('<div class="section-title">👋 About Me</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            Hi! I’m <span class="highlight">Muhammad Umer Khan</span>, an aspiring AI/Data Scientist passionate about 
            <span class="highlight">🤖 Natural Language Processing (NLP)</span> and 🧠 Machine Learning. 
            Currently pursuing my Bachelor’s in Computer Science, I bring hands-on experience in developing intelligent recommendation systems, 
            performing data analysis, and building machine learning models. 🚀
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎯 Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            Here are some of the key projects I have worked on:
            <ul>
                <li><span class="highlight">📋 Description:</span> 
                    Pneumonia🩻 is a life-threatening infectious disease affecting one or both lungs in humans commonly caused by bacteria called Streptococcus pneumoniae. 
                    One in three deaths in world is caused due to pneumonia as reported by World Health Organization (WHO). 
                    Chest X-Rays which are used to diagnose pneumonia need expert radiotherapists for evaluation. 
                    Thus, developing an automatic system for detecting pneumonia would be beneficial for treating the 
                    disease without any delay particularly in remote areas. Due to the success of deep learning algorithms 
                    in analyzing medical images, Convolutional Neural Networks (CNNs) have gained much attention for disease 
                    classification. In addition, features learned by pre-trained CNN models on large-scale datasets are much 
                    useful in image classification tasks. In this work, we appraise the functionality of pre-trained CNN models 
                    utilized as feature-extractors followed by different classifiers for the classification of abnormal and 
                    normal chest X-Rays. We analytically determine the optimal CNN model for the purpose. 
                    Statistical results obtained demonstrates that pretrained CNN models employed along with supervised classifier algorithms can be very beneficial in analyzing chest X-ray images, 
                    specifically to detect Pneumonia.<br/>
                </li>
                <li><span class="highlight">🩻 Pneumonia Disease Detection:</span> 
                    Built disease detection 🦠 model to identify diseases using pre-trained Convolutional Neural Networks (CNNs) 🧠. 
                    The model was trained on a 
                    <a href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia" target="_blank" style="color: silver; font-weight: bold;">dataset</a>
                    of x-ray report images and is deployed for real-time predictions. 📡<br/>
                    <ul>
                        <li><span class="highlight">🤝 Steps to Reproduce:</span>
                            <ul>
                                <li>Captured in a real potato farm.</li>
                                <li>Uncontrolled environment using a high-resolution digital camera and smartphone.</li>
                                <li>Dataset aids researchers in computer vision.</li>
                            </ul>
                        </li>
                        <li><span class="highlight">🔄 Data Preprocessing and Augmentation:</span>
                            <ul>
                                <li><span class="highlight">Image Cleaning:</span>
                                    <ul>
                                        <li>Removing noise, artifacts, and unwanted objects.</li>
                                    </ul>
                                </li>
                                <li><span class="highlight">Image Resizing:</span>
                                    <ul>
                                        <li>Converting images to a consistent size for efficient processing.</li>
                                    </ul>
                                </li>
                                <li><span class="highlight">Image Normalization:</span>
                                    <ul>
                                        <li>Adjusting pixel values to a specific range (e.g., 0-1).</li>
                                    </ul>
                                </li>
                                <li><span class="highlight">Data Augmentation:</span>
                                    <ul>
                                        <li>Creating new training samples by applying transformations:</li>
                                        <ul>
                                            <li>Rotation 🔄</li>
                                            <li>Flipping 🔁</li>
                                        </ul>
                                    </ul>
                                </li>
                            </ul>
                        </li>
                    </ul>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html=True)


    # Future Work Section
    # st.markdown('<div class="section-title">🚀 Future Work</div>', unsafe_allow_html=True)
    # st.markdown("""
    #     <div class="content">
    #         While this project currently focuses on potato plant disease classification, I aim to expand its scope to cover:
    #         <ul>
    #             <li><span class="highlight">🌾 Multi-Crop Disease Detection:</span> Incorporating classification models for other crops like tomatoes, wheat, and corn.</li>
    #             <li><span class="highlight">🤝 Farmer-Friendly Mobile App:</span> Developing a user-friendly mobile application to enable real-time field diagnosis and recommendations for farmers.</li>
    #         </ul>
    #         These enhancements aim to provide a comprehensive tool for farmers and agricultural researchers, contributing to sustainable farming practices. 🌱
    #     </div>
    # """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">💻 Technologies & Tools</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <ul>
                <li><span class="highlight">🔤 Languages & Libraries:</span> Python, NumPy, Pandas, Matplotlib, TensorFlow, Keras, and Scikit-Learn.</li>
                <li><span class="highlight">⚙️ Approaches:</span> Pre-trained CNNs, Data Augmentation, Transfer Learning, and Image Preprocessing Techniques.</li>
                <li><span class="highlight">🌐 Deployment:</span> Streamlit for building an interactive, user-friendly web-based system.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-title">🩻 Pneumonia Disease Detector 🦠</div>', unsafe_allow_html=True)
    st.markdown('''
    <div class="content">
        Upload a clear image of your x-ray report 🩻, and the model will identify its health status or diagnose any potential disease as following:
        <ul>
            <li>Healthy 💪.</li>
            <li>Pneumonia Defected 🦠.</li>
        </ul>
    </div><br/>
    ''', unsafe_allow_html=True)

    # Layout with two columns
    col1, col2 = st.columns([1, 2])  # 1: Image section, 2: Prediction section

    with col1:
        uploaded_file = st.file_uploader("📸 Upload x-ray report:", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            image = image.convert("RGB")  # Ensure the image has 3 channels (RGB)
            st.image(image, caption="Uploaded Image", width=250)

    with col2:
        if uploaded_file is not None:
            # Preprocess the image
            image = image.resize((150, 150))  # Resize as per model input
            image_array = np.array(image) / 255.0  # Normalize to [0, 1]
            image_batch = np.expand_dims(image_array, axis=0)  # Add batch dimension

            # Load the model
            pneumonia_classifier_model = tf.keras.models.load_model('./model/pnemonia_classifier_v2.h5')
            class_names = ["Normal", "Pneumonia"]

            # Make predictions
            predictions = pneumonia_classifier_model.predict(image_batch)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])

            # Display the prediction result
            if predicted_class.lower() == "normal":
                status_message = f"Your 🩻 Report is <span style='color: #4CAF50;'>{predicted_class}</span> 💪."
            else:
                status_message = f"Damn! You are <span style='color: #c40000;'>{predicted_class}</span> defected 🦠."

            st.markdown(f'''
            <br/><br/><br/><br/><br/><br/><br/><br/><br/>
            <div style="font-size: 2em; font-weight: bold; text-align: center;">
                {status_message}
            </div>
            ''', unsafe_allow_html=True)

# Footer Section
st.markdown("""
    <div class="footer">
        Developed by <a href="https://portfolio-sigma-mocha-67.vercel.app" target="_blank" style="color: silver; font-weight: bold;">Muhammad Umer Khan</a>. Powered by TensorFlow and Streamlit. 🌐
    </div>
""", unsafe_allow_html=True)