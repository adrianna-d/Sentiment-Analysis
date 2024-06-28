# Sentiment-Analysis

![twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)
![Reddit User Karma](https://img.shields.io/reddit/user-karma/combined/badge?style=for-the-badge&logo=Reddit&logoColor=%23FF5700)
![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![license-shield](https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge)
![linkedin-shield](https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555)


<!-- DESCRIPTION -->
## Description

The Social Media Emotions Analysis Tool is an application that allows users to input a sentence or keyword and receive real-time analysis of sentiment and emotions associated with it across social media platforms. This tool enhances decision-making processes, improves marketing strategies, and provides actionable insights based on sentiment and emotional responses across various social media platforms. In **current development** there is a feature of the input being measured based on a chosen social media platform and data from last 7 days.

This project explores  **emotion detection** along with **text preprocessing** and **feature engineering methods** using **BERT** model.
It is end-to-end pipeline that provides insights about any topic or product through real-time tweets sentiment analysis and emotion detection.
Final project for the **Data Science and Machine Learning**  bootcamp at IronHack.

Currently pulling real-time data from Twitter, and predicting emotions to generate insights in a local file with the goal to move it to the app.

## Files

The project consists of the following files:

- [Tableau Dashboard](https://public.tableau.com/shared/CTPSRXFT6?:display_count=n&:origin=viz_share_link): On overview of data across Facebook,Instagram and Twitter from [sentimentdataset.csv](data\sentimentdataset.csv)

- [model_code.py](/model_code.py): Contains functions for loading and saving model, vectorizer, data, and performing predictions.

- [predictions_json.py](/predictions_json.py): Code runs the model, input your json file with text to make predictions.
   
- [ML_codes_all.ipynb](/ML_codes_all.ipynb):Codes for different trained models, transformers, label_encoders.
   
- [requirements.txt](/requirements.txt): Lists all the packages required for the project.
- [datacleaning.sql](datacleaning.sql): Data cleaning, preprocessing and saving. Very similar to [datapreparation.py](data_preparation.py) which I also included.

- [data](/data): A data folder containing datasets used for model training.
    * You can use val_data folder to test your code as well, the data there is not in the combined_emotions.csv.
    * [combined_emotions.csv](data/combined_emotions.csv): dataset used for model training.
    * The file the combined_emotions.csv contains approx 2.1 mln online posts and assosciated emotions with them.
    * Data was downloaded from the below links, as it is too big I am not adding it to the repo, you can find the way to combine and criate the training .csv in [datapreparation.py](data_preparation.py) :

        * https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset
        * https://www.kaggle.com/datasets/kazanova/sentiment140
        * https://www.kaggle.com/datasets/parulpandey/emotion-dataset
        * https://www.kaggle.com/datasets/bhavikjikadara/emotions-dataset
        * https://huggingface.co/datasets/dair-ai/emotion
        * https://www.kaggle.com/datasets/chandrug/textemotiondetection?select=val.txt
        * https://github.com/SannketNikam/Emotion-Detection-in-Text/blob/main/data/emotion_dataset.csv
        * Pre-labeled sentiment dataset Sentiment140



## Getting Started

To run the project, follow the steps below:

1. Clone the repository:

    ```bash
    git clone https://github.com/adrianna-d/Sentiment-Analysis.git
    ```

2. Optional but encougraged: Create a virtual environment for the project:
    
    ```bash
    python -m venv venv
    venv/Scripts/activate # or source venv/bin/activate
    ```
3. Install the required packages listed in [requirements.txt](requirements.txt):
    
    ```bash
    pip install -r requirements.txt
    ```
   
4. Optional, if using APIs save your credentials in a safe place in a .yaml or .cfg file ;) 

5. Run the model_code.py to create your model, tokenizer, label_encoder.

6. Runn the predictions_json.py with your json file to precidt emotions of taxt from the file.

6. Run the app.py to with predictions locally

<!-- PROJECT OVERVIEW: METHODOLOGY AND APPROACH -->
## Project overview: Methodology and Approach

### Project Pipeline
- **Data Collection**: Collect datasets from kaggle. Collect tweets and posts from Twitter and Reddit using respective APIs.
- **Data Pre-processing**: Clean and preprocess the data, tokenize using BERT tokenizer.
- **Model Training**: Train the BERT model on the preprocessed data for emotion detection.
- **Model Evaluation**: Evaluate the model performance using appropriate metrics.
- **Real-time Data Fetching**: Fetch real-time data from Twitter and Reddit.
- **Prediction and Reporting**: Predict emotions and generate reports.

### Data Pre-processing and Feature Engineering

The data pre-processing step includes the following steps, data was preprocessed in SQL before.
- **Data Cleaning**: Remove Nan values.
- **Tokenization**: Convert text to token IDs using BERT tokenizer.
- **Padding and Truncation**: Ensure uniform input length for the BERT model.

### Model Training and Evaluation

The model training and evaluation step includes the following steps:
- **Train/Test Split**: Split the data into training and testing sets.
- **Model Training**: Train the model using the training set.
- **Model Evaluation**: Evaluate the model using the testing set.
- **Metrics**: Use accuracy, precision, recall, F1-score to evaluate the model.
- **Confusion Matrix**: Generate a confusion matrix to evaluate the model's performance.

### Real-time Data Fetching

- **Twitter API**: Use the Twitter API to fetch real-time tweets (paid)
- **Tweepy**: Use the `tweepy` package to fetch tweets from the Twitter API (you stil need to pay to X)
- **WebScrapping**: use webstrapping like Apify
- **Data Pre-processing**: Pre-process the fetched tweets using the same steps as well as the same vectorizer as the data pre-processing step.

### Prediction

- **Prediction**: Predict the sentiment and emotions of the fetched comments/text using the trained BERT model.
  
### Logging and Model Management with MLflow
- **Experiment Tracking**: Track experiments, parameters, and metrics using MLflow.
- **Model Logging**: Log the trained BERT model, tokenizer, and label encoder with MLflow.
- **Artifacts**: Save and log artifacts like the classification report, model files, and tokenizer.

## References

- Images used for the app were created by rawpixel.com for [Freepik](https://www.freepik.com/author/rawpixel-com).
- Big thanks to all Kaggle users!

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
