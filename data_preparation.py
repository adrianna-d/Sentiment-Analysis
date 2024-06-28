import pandas as pd
# datasets
data1 = pd.read_csv("data/emotion_dataset.csv")
#joy, sadness, fear, anger, surprise, neutral, disgust, shame ( 34795 rows)
data2 = pd.read_csv("data/test_01.txt", sep=";", header=None, names=["text", "emotion"])
# 'anger','sadness','fear','joy','surprise','love'
data3 = pd.read_csv("data/test.csv")
# sadness (0), joy (1), love (2), anger (3), fear (4) (2000 tweets)
data4 = pd.read_csv("data/training.csv")
# sadness (0), joy (1), love (2), anger (3), fear (4) (16000 tweets)
data5 = pd.read_csv("data/emotions.csv")
#sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5).(400k + rows)
data6 = pd.read_csv(r"data\tweet_emotions.csv")
#'empty' 'sadness' 'enthusiasm' 'neutral' 'worry' 'surprise' 'love' 'fun' 'hate' 'happiness' 'boredom' 'relief' 'anger'
data7 = pd.read_csv("data\sentimentdataset.csv")
# lots
data8 = pd.read_csv(r"data\training_noemotnicon.csv", encoding='latin1')
# 4=positive 0= negative 2=neutral

# Define the emotion mapping with numeric labels
emotion_mapping = {
    'sadness': 0,
    'joy': 1,
    'love': 2,
    'anger': 3,
    'fear': 4,
    'surprise': 5,
    'neutral': 6,
    'disgust': 7,
    'shame': 8,
    'enthusiasm': 9,
    'worry': 10,
    'fun': 11,
    'hate': 12,
    'happiness': 13,
    'boredom': 14,
    'relief': 15

}
# Create a reverse mapping dictionary
reverse_emotion_mapping = {v: k for k, v in emotion_mapping.items()}

# Function to map numeric labels to text emotions
def map_emotion(df, col):
    df[col] = df[col].map(reverse_emotion_mapping)
    return df

# Keep only 'Emotion' and 'Text' columns
data1 = data1[['Emotion', 'Text']]
data1.columns = ['emotion', 'text']
data1.to_csv("data/processed_data1.csv", index=False)

data2.columns = ['text', 'emotion']
data2.to_csv("data/processed_data2.csv", index=False)

data3 = data3.rename(columns={ 'label': 'emotion'})
data3 = map_emotion(data3, 'emotion')
data3.to_csv("data/processed_data3.csv", index=False)

data4 = data4.rename(columns={ 'label': 'emotion'})
data4 = map_emotion(data4, 'emotion')
data4.to_csv("data/processed_data4.csv", index=False)

data5 = data5.rename(columns={ 'label': 'emotion'})
data5 = map_emotion(data5, 'emotion')
data5.to_csv("data/processed_data5.csv", index=False)

data6 = data6.rename(columns={'content': 'text', 'sentiment': 'emotion'})
data6 = data6[['emotion', 'text']]
data6.to_csv("data/processed_data6.csv", index=False)

data7 = data7.rename(columns={ 'Sentiment': 'emotion', 'Text': 'text'})
data7 = data7[['emotion', 'text']]
data7.to_csv("data/processed_data7.csv", index=False)

data8.columns = ['emotion', 'tweet_id', 'date', 'query', 'user', 'text']
sentiment_mapping_v1 = {
    0: 'negative',
    2: 'neutral',
    4: 'positive'
}
# Function to map numeric sentiments to text labels
def map_sentiment_to_text_v1(df, col):
    df[col] = df[col].map(sentiment_mapping_v1)
    return df
data8 = map_sentiment_to_text_v1(data8, 'emotion')
data8 = data8[['emotion', 'text']]
data8.to_csv("data/processed_data8.csv", index=False)
# List of file paths CSV 
csv_files = [
    "data/processed_data1.csv",
    "data/processed_data2.csv",
    "data/processed_data3.csv",
    "data/processed_data4.csv",
    "data/processed_data5.csv",
    "data/processed_data6.csv",
    "data/processed_data7.csv",
    "data/processed_data8.csv"
]

# List to store all DataFrames
dfs = []

# Load each CSV into a DataFrame
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all DataFrames into one
combined_data = pd.concat(dfs, ignore_index=True)

# Number of rows to sample
num_rows_to_sample = 20000

# Check if number of rows to sample is greater than total rows available
if num_rows_to_sample > len(combined_data):
    raise ValueError(f"Number of rows to sample ({num_rows_to_sample}) is greater than total rows available in the combined dataset ({len(combined_data)}).")

# Sample 20,000 random rows from the combined dataset
random_indices = np.random.choice(len(combined_data), size=num_rows_to_sample, replace=False)
df_sampled = combined_data.iloc[random_indices]

# Save the sampled data to model_eval.csv
model_eval_file = 'model_eval.csv'
df_sampled.to_csv(model_eval_file, index=False)
print(f"Successfully saved {num_rows_to_sample} randomly sampled rows to {model_eval_file}.")

# Remove the sampled rows from combined_data to get the remaining rows
remaining_data = combined_data.drop(index=df_sampled.index)

# Save the remaining data to combined_emotions.csv
combined_emotions_file = 'combined_emotions.csv'
remaining_data.to_csv(combined_emotions_file, index=False)
print(f"Successfully saved the remaining {len(remaining_data)} rows to {combined_emotions_file}.")
