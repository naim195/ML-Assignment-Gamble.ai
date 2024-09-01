# Human Activity Recognition (HAR) Assignment

This project focuses on classifying human activities based on accelerometer data using various machine learning techniques. The assignment consists of four main tasks: Exploratory Data Analysis (EDA), Decision Tree Implementation, Prompt Engineering for Large Language Models (LLMs), and Data Collection in the Wild. The following sections summarize the work done for each task.

## Exploratory Data Analysis (EDA)

1. **Preprocessing**: 
   - Combined and organized the accelerometer data from the UCI-HAR dataset using provided scripts (`CombineScript.py` and `MakeDataset.py`).
   - Focused on the first 10 seconds of activity data, resulting in 500 samples at a 50Hz sampling rate.

2. **Waveform Plotting**:
   - Plotted waveforms for one sample data from each activity class to observe differences and similarities between activities.
   - Subplots were created for all six activities (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying).

3. **Linear Acceleration Analysis**:
   - Analyzed the linear acceleration (sum of squared acceleration components) to distinguish between static and dynamic activities.
   - Concluded whether machine learning models are necessary for differentiating between static and dynamic activities.

4. **PCA Visualization**:
   - Performed PCA on total acceleration data and visualized different classes of activities through scatter plots.
   - Applied TSFEL to generate features and visualized them using PCA.
   - Compared PCA results from total acceleration, TSFEL features, and features provided by the dataset.

5. **Correlation Analysis**:
   - Calculated the correlation matrix of features from TSFEL and the dataset.
   - Identified highly correlated features and evaluated redundancy.

## Decision Trees for Human Activity Recognition

1. **Model Training**:
   - Trained Decision Tree models using raw accelerometer data, TSFEL features, and dataset-provided features.
   - Evaluated models using accuracy, precision, recall, and confusion matrix.

2. **Model Comparison**:
   - Compared the performance of Decision Trees trained on different feature sets.
   - Analyzed model performance on participants/activities where results were suboptimal.

3. **Depth Variation**:
   - Trained Decision Trees with varying depths (2-8) and plotted the accuracy on test data against tree depth.

## Prompt Engineering for Large Language Models (LLMs)

1. **Zero-Shot and Few-Shot Learning**:
   - Demonstrated Zero-Shot and Few-Shot Learning for classifying human activities using featurized accelerometer data.
   - Qualitatively compared the performance of Few-Shot Learning with Zero-Shot Learning.

2. **Quantitative Comparison**:
   - Compared the accuracy of Few-Shot Learning with Decision Trees.
   - Discussed the limitations of Zero-Shot and Few-Shot Learning in human activity classification.

3. **Model Testing**:
   - Evaluated the model’s response to new and random activity data to understand its generalization capabilities.

## Data Collection in the Wild

1. **Data Collection**:
   - Collected accelerometer data for various activities using a smartphone app.
   - Processed and trimmed the data to obtain 10-second samples for each activity.

2. **Model Testing**:
   - Tested the Decision Tree model trained on the UCI-HAR dataset with collected data.
   - Applied preprocessing and featurization as needed to enhance model performance.
   - Tested Few-Shot prompting on both UCI-HAR and collected data.

## Decision Tree Implementation

1. **Decision Tree Algorithm**:
   - Implemented a Decision Tree model from scratch in Python, capable of handling different types of features and outputs.
   - Implemented Information Gain using Entropy, Gini Index, and MSE for splitting.

2. **Performance Metrics**:
   - Evaluated the Decision Tree model on a synthetic dataset using accuracy, precision, and recall.
   - Performed 5-fold cross-validation and nested cross-validation to find the optimal tree depth.

3. **Automotive Efficiency**:
   - Applied the Decision Tree model to the automotive efficiency problem.
   - Compared the performance of the custom Decision Tree model with the Scikit-learn implementation.

4. **Runtime Complexity Analysis**:
   - Conducted experiments on runtime complexity by varying the number of samples (N) and features (M) in the dataset.
   - Analyzed and compared the time complexity for tree creation and prediction with theoretical expectations.
