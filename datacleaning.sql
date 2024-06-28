CREATE DATABASE IF NOT EXISTS emotions_db;

USE emotions_db;

CREATE TABLE IF NOT EXISTS emotion_data1 (
    Emotion VARCHAR(255),
    Text TEXT
);

CREATE TABLE IF NOT EXISTS emotion_data2 (
    Text TEXT,
    Emotion VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS emotion_data3 (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT,
    label INT
);

CREATE TABLE IF NOT EXISTS emotion_data4 (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT,
    label INT
);

CREATE TABLE IF NOT EXISTS emotion_data5 (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT,
    label INT
);

CREATE TABLE IF NOT EXISTS emotion_data6 (
    tweet_id INT,
    sentiment VARCHAR(255),
    content TEXT
);

CREATE TABLE IF NOT EXISTS emotion_data7 (
    Sentiment VARCHAR(255),
    Text TEXT,
    Unnamed VARCHAR(255),
    Timestamp VARCHAR(255),
    User VARCHAR(255),
    Platform VARCHAR(255),
    Hashtags VARCHAR(255),
    Retweets INT,
    Likes INT,
    Country VARCHAR(255),
    Year INT,
    Month INT,
    Day INT,
    Hour INT
);

CREATE TABLE IF NOT EXISTS emotion_data8 (
    emotion INT,
    tweet_id INT,
    date DATE,
    query VARCHAR(255),
    user VARCHAR(255),
    text TEXT
);

LOAD DATA INFILE 'C:/mysql-files/emotion_dataset.csv'
INTO TABLE emotion_data1
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(Emotion, Text);

LOAD DATA INFILE 'C:/mysql-files/test.csv'
INTO TABLE emotion_data3
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(text, label);

LOAD DATA INFILE 'C:/mysql-files/training.csv'
INTO TABLE emotion_data4
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(text, label);

LOAD DATA INFILE 'C:/mysql-files/emotions.csv'
INTO TABLE emotion_data5
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(text, label);

LOAD DATA INFILE 'C:/mysql-files/tweet_emotions.csv'
INTO TABLE emotion_data6
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(tweet_id, sentiment, content);

LOAD DATA INFILE 'C:/mysql-files/sentimentdataset.csv'
INTO TABLE emotion_data7
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(Unnamed, Text, Sentiment, Timestamp, User, Platform, Hashtags, Retweets, Likes, Country, Year, Month, Day, Hour);

LOAD DATA INFILE 'C:/mysql-files/training_noemotnicon.csv'
INTO TABLE emotion_data8
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(emotion, tweet_id, date, query, user, text);

-- Process emotion_data3
CREATE TABLE IF NOT EXISTS processed_emotion_data3 AS
SELECT 
    id,
    text,
    CASE label
        WHEN 0 THEN 'sadness'
        WHEN 1 THEN 'joy'
        WHEN 2 THEN 'love'
        WHEN 3 THEN 'anger'
        WHEN 4 THEN 'fear'
        ELSE CAST(label AS CHAR)
    END AS emotion
FROM emotion_data3;

-- Process emotion_data4
CREATE TABLE IF NOT EXISTS processed_emotion_data4 AS
SELECT 
    id,
    text,
    CASE label
        WHEN 0 THEN 'sadness'
        WHEN 1 THEN 'joy'
        WHEN 2 THEN 'love'
        WHEN 3 THEN 'anger'
        WHEN 4 THEN 'fear'
        ELSE CAST(label AS CHAR)
    END AS emotion
FROM emotion_data4;

-- Process emotion_data5
CREATE TABLE IF NOT EXISTS processed_emotion_data5 AS
SELECT 
    id,
    text,
    CASE label
        WHEN 0 THEN 'sadness'
        WHEN 1 THEN 'joy'
        WHEN 2 THEN 'love'
        WHEN 3 THEN 'anger'
        WHEN 4 THEN 'fear'
        WHEN 5 THEN 'surprise'
        ELSE CAST(label AS CHAR)
    END AS emotion
FROM emotion_data5;

-- Process emotion_data8
CREATE TABLE IF NOT EXISTS processed_emotion_data8 AS
SELECT 
    tweet_id,
    CASE emotion
        WHEN 0 THEN 'negative'
        WHEN 2 THEN 'neutral'
        WHEN 4 THEN 'positive'
        ELSE CAST(emotion AS CHAR)
    END AS emotion,
    text
FROM emotion_data8;

-- Combine all tables into one
CREATE TABLE IF NOT EXISTS combined_emotions AS
SELECT Emotion AS emotion, Text AS text FROM emotion_data1
UNION ALL
SELECT Emotion AS emotion, Text AS text FROM emotion_data2
UNION ALL
SELECT emotion, text FROM processed_emotion_data3
UNION ALL
SELECT emotion, text FROM processed_emotion_data4
UNION ALL
SELECT emotion, text FROM processed_emotion_data5
UNION ALL
SELECT sentiment AS emotion, content AS text FROM emotion_data6
UNION ALL
SELECT Sentiment AS emotion, Text AS text FROM emotion_data7
UNION ALL
SELECT emotion, text FROM processed_emotion_data8;

-- Add an ID column to facilitate random sampling
ALTER TABLE combined_emotions ADD COLUMN id INT AUTO_INCREMENT PRIMARY KEY;

-- Sample 20,000 rows randomly
CREATE TABLE IF NOT EXISTS model_eval AS
SELECT emotion, text
FROM combined_emotions
ORDER BY RAND()
LIMIT 20000;

-- Save the remaining rows
CREATE TABLE IF NOT EXISTS combined_emotions_remaining AS
SELECT emotion, text
FROM combined_emotions
WHERE id NOT IN (SELECT id FROM model_eval);
