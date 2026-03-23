# Social Media Engagement & Sentiment Analysis

This repository contains a dual-model approach to analyzing social media data. It features a **Forecast Model** to predict engagement metrics and a **Sentiment & Emotion Predictor** that leverages Deep Learning to understand the emotional context of post content.

## Features

* **Engagement Forecasting:** Predicts `engagement_rate` using various regression models, including Random Forest, SVR, and XGBoost. 
* **Deep Learning Sentiment Analysis:** Utilizes `DistilBERT` for high-performance text embedding and sentiment/emotion classification. 
* **Comprehensive Data Pipeline:** Includes dedicated stages for data cleaning, outlier removal (addressing high-kurtosis columns like engagement rate), and statistical profiling. 
* **Feature Engineering:** Extracts insights from `hashtags`, `mentions`, and `topic_categories` to improve predictive accuracy. 

## Project Structure

* `forecast_model_predictor.ipynb`: Notebook focused on regression analysis of engagement metrics (likes, shares, comments). 
* `sentiment_emotion_predictor_code.ipynb`: PyTorch-based notebook for NLP tasks using Transformers. 
* `requirements.txt`: Core dependencies for the project environment.

## Prerequisites

The project requires Python 3.x and several data science libraries. Key dependencies include:
* **Data Processing:** `pandas`, `numpy`, `scipy`
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Deep Learning:** `torch` (PyTorch), `transformers` (HuggingFace)
* **Visualization:** `matplotlib`, `seaborn`

## Installation

1. Clone the repository

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For GPU acceleration with PyTorch, follow the instructions in the `requirements.txt` file for specific CUDA versions.*

## Data Insights

The analysis is performed on a **Social Media Engagement Dataset**. Key data processing steps include:
* **Outlier Management:** Handling highly skewed distributions, particularly in `engagement_rate` which can show extreme values (Leptokurtic distribution).
* **Cleaning:** Addressing missing values in columns like `mentions`. 
* **Exploration:** Statistical profiling of numeric data to understand user behavior patterns (e.g., comparing share rates vs. comment rates).

## Models Used

* **Regression:** Decision Trees, Random Forest, SVR, and XGBoost for numerical predictions. 
* **NLP:** DistilBERT for text content analysis and classification. 
