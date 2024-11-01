# Real-Time Emotion-Based Music Recommendation System 🎶😊😢😡

This project is a real-time music recommendation system that detects a user’s emotions using facial recognition and plays music that aligns with their current mood. By analyzing emotions like happiness, sadness, anger, and calmness, it enhances the listening experience by suggesting songs to match the listener's mood.

## 🎯 Project Objective
The primary goal of this project is to leverage computer vision and machine learning to recommend songs based on real-time emotion detection. It’s designed to:
- Detect emotions from facial expressions via webcam input.
- Analyze detected emotions to recommend mood-based music.
- Provide users with a personalized listening experience.

## ✨ Features
- **Real-Time Emotion Detection:** Uses a camera feed to recognize emotions in real-time.
- **Mood-Based Music Recommendation:** Recommends and plays songs matching detected emotions.
- **Seamless Spotify Integration:** Retrieves mood-based music suggestions via the Spotify API.
- **User-Friendly Interface:** A clean, intuitive interface to enhance user interaction.

## 💻 Tech Stack
- **Programming Language:** Python
- **Machine Learning Frameworks:** TensorFlow or PyTorch for emotion detection
- **Computer Vision:** OpenCV for face tracking and image processing
- **Spotify API:** `spotipy` for music recommendations
- **Other Libraries:** `dlib`, `scikit-learn`, `pandas
## 🚀 Installation
1. **Clone the repository**:
git clone https://github.com/Harry-2898/Music.git
   cd EmotionMusicRecSystem
python src/main.py
Interact with the System:

Allow access to your webcam when prompted.
The system will detect your emotion and play songs based on your mood.
Adjust Model and Song Preferences:

Customize emotion categories and song genres in the music_recommender.py file as per your preferences.
📊 Training and Testing the Model
The src/emotion_detector.py file contains the code for training the emotion detection model. To train on a new dataset:

Place the dataset in the data/ directory.
Adjust the paths and parameters in detectEmotionRealTime.py.
Run the script to train and save a new model.
🤝 Contributing
Feel free to open an issue or submit a pull request for any improvements. Contributions are always welcome!

🛠 Future Improvements
Expand Emotion Categories: Increase the variety of emotions detected for a more nuanced music experience.
Additional Music APIs: Integrate more music platforms (e.g., Apple Music, YouTube) to expand song recommendations.
Advanced Emotion Detection: Enhance model accuracy with more complex architectures or multi-modal inputs.


Happy coding and listening! 🎧
