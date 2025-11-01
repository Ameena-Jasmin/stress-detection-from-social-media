# Stress Detection from Social Media Posts

This project uses Natural Language Processing (NLP) and Machine Learning to detect whether a social media post expresses **stress** or **non-stress** emotions.

## Features
- Text preprocessing using NLTK (tokenization, stopword removal, etc.)
- TF-IDF feature extraction
- Logistic Regression classifier
- Streamlit web app for real-time predictions

## Model Performance
**Accuracy:** 86.8%  
**Precision (Stress):** 0.86  
**Recall (Stress):** 0.92

## Example
| Input Text | Prediction |
|-------------|-------------|
| "Feeling overwhelmed with exams" | Stress |
| "Just finished a workout, feeling great!" | Non-Stress |

## How to Run Locally
```bash
git clone https://github.com/Ameena-Jasmin/stress-detection-from-social-media.git
cd stress-detection-from-social-media
pip install -r requirements.txt
streamlit run app.py
