# 🎶 Emotion-Based Music Recommender  

> *“Let your face choose your playlist.”*  
An AI-powered system that detects emotions from facial expressions and recommends music to match your mood.  

---

![Emotion Music Banner](https://img.shields.io/badge/Emotion%20AI-Music%20Recommender-blueviolet?style=for-the-badge&logo=tensorflow&logoColor=white)  

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=flat&logo=tensorflow)  
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=flat&logo=opencv)  

---

## ✨ Features  

- 🧠 **Emotion Detection** → Detects emotions (*happy, sad, angry, surprised, neutral, etc.*) from faces.  
- 🎵 **Music Recommendation** → Maps emotions to curated playlists/songs.  
- ⚡ **Pre-Trained CNN Model** → Trained on **FER-2013 dataset**.  
- 🖼️ **Modular Design** → Easy to extend with new datasets or APIs (Spotify, YouTube).  
- 📸 **Real-time Ready** → Works with static images or webcam feed.  

---

## 🗂️ Project Structure  

Emotion_based_Music_recommender/
├── scripts/ # Training & inference scripts
├── model/ # Pre-trained model
├── balanced_data/ # Processed dataset
├── emojis/ # Emoji assets for visualization
├── fer2013/ # FER-2013 dataset
├── requirements.txt # Dependencies
└── README.md


---

## 🚀 Getting Started  

### 1️⃣ Clone & Install  
```bash
git clone https://github.com/Baishakhi-Sing/Emotion_based_Music_recommender.git
cd Emotion_based_Music_recommender
pip install -r requirements.txt

2️⃣ Download Dataset & Model

📥 FER-2013 dataset → Place in fer2013/

📥 Pre-trained model → Place in model/

3️⃣ Run the App
python scripts/run.py


🛠️ Tech Stack

Python 3.8+

TensorFlow / Keras → Emotion detection CNN

OpenCV → Face capture & processing

NumPy, Pandas, Matplotlib → Data handling & visualization


## Future Roadmap

🎧 Spotify/YouTube API integration (stream songs directly).

💻 GUI Application (Tkinter / PyQt).

📹 Real-time webcam emotion detection.


## Authors

Srizoni Maity 
LinkedIn : [https://www.linkedin.com/in/srizoni-maity-012235356]

📧 Contact: [im.srizoni@gmail.com]


🙌 Acknowledgments

FER-2013 dataset for training.

TensorFlow & Keras docs.

Inspiration from the idea of connecting AI + Human emotions + Music.
