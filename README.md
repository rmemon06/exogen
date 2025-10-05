# Exogen -  AI exoplanet model
**My submission for the NASA Space Apps Challenge - Rohan Memon**

Exogen is an interactive web app in which you can use my machine learning model built on the Kepler Objects of interest database. You can use it to determine whether or not a KOI is a planet, candidate or not with 92% accuracy. My goal is to make this easy to use and accessible for anyone to learn and engage with.

**Public link to the video:** https://youtu.be/pUpi5-xHSig

## Key features:
- Single object prediction: Users can enter the Kepler ID of any KOI to receive instant analysis comparing the prediction and the NASA classification of it.
- Batch CSV analyis: Apart from just classifying already found KOIs, you can upload your own CSV file containing as many objects as you would like, and recieve the models classification of each
- Model transparency: No machine learning model is 100% accurate, therefore we have a data and analytics page which visualises the models performance, including its accuracy, the confusion matrix and a chart of its most important features
- Projects vision: I wanted to do a lot more with this and therefore added a Roadmap page to outline my future plans!

## How to run this project!
1. Prerequistes: Make sure you have python and the following libraries: pip install pandas scikit-learn flask matplotlib seaborn joblib
2. Download the files: app.py is the main file and the dataset also
3. Run the app: Within the terminal fun python app.py
4. Access: Open the web browser and go to http://127.0.0.1:5000

## Nasa Resources Used:
The project is built upon the Kepler Objects of Interest dataset provided by the NASA exoplanet archive.
I have future plans to upgrade the model and train it also on the TESS, K2 data.
   
