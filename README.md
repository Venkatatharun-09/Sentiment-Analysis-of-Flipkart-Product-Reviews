Flipkart Review Sentiment Analysis with MLflow
This project builds a Machine Learning pipeline to classify Flipkart product reviews as Positive or Negative using Natural Language Processing (NLP) techniques.
All experiments are tracked and managed using MLflow, and the results are shown through an interactive application.

Project Overview
CCustomer reviews are an important source of feedback for e-commerce platforms.
This project analyzes review text and predicts the sentiment of a review based on its rating.

Sentiment Rules
• Ratings ≥ 4 → Positive
• Ratings ≤ 2 → Negative
• wRating = 3 (Neutral) → Removed from the dataset
The model uses TF-IDF Vectorization and Logistic Regression for sentiment classification.

Objectives
• Perform sentiment classification on Flipkart product reviews
• Build a complete Machine Learning pipeline
• Track experiments using MLflow
• Log model parameters, metrics, and artifacts
• Register the trained model for reuse and deployment

Tech Stack
• Python
• Pandas
• Scikit-learn
• MLflow
• Matplotlib

Project Workflow
• Load and clean the dataset
• Convert product ratings into sentiment labels
• Split the data into training and testing sets
• Create a Machine Learning pipeline:
   TF-IDF Vectorizer
   Logistic Regression Classifier
• Train the model
• Evaluate the model using:
   Accuracy
   F1 Score
• Track experiments using MLflow
• Save the confusion matrix as an artifact
• Register the trained model for future use

MLflow Experiment Tracking
The following information is tracked using MLflow:
Model Parameters:
  Regularization parameter (C)
  Maximum iterations (max_iter)
  Maximum TF-IDF features (max_features)
Evaluation Metrics:
  Accuracy
  F1 Score
Artifacts:
  Confusion Matrix
Registered Model:
 FlipkartSentimentClassifier
