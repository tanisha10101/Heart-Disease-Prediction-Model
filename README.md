# Heart Disease Prediction Model

This project aims to predict the likelihood of heart disease in patients based on various medical features. Heart disease is one of the leading causes of death worldwide, and early detection can save lives by enabling preventive actions. This machine learning project leverages popular classification algorithms to predict whether a patient is likely to develop heart disease based on their medical history and lifestyle factors.

---

## Dataset

The dataset used for this project can be found at the following link:

- **<https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset>)**

The dataset used for this prediction includes several medical features that are commonly associated with heart disease. These features include:

- **Age**: The age of the patient.
- **Sex**: The gender of the patient.
- **Blood Pressure**: Systolic and diastolic blood pressure readings.
- **Cholesterol Level**: Total cholesterol level in the blood.
- **Fasting Blood Sugar**: Whether the patient has a fasting blood sugar greater than 120 mg/dl.
- **Resting Electrocardiographic Results**: Electrocardiogram (ECG) results.
- **Maximum Heart Rate Achieved**: The highest heart rate achieved during exercise.
- **Exercise Induced Angina**: Whether the patient experiences chest pain during exercise.
- **Oldpeak**: Depression induced by exercise relative to rest.
- **Slope of Peak Exercise ST Segment**: The slope of the ST segment during peak exercise.
- **Number of Major Vessels Colored by Fluoroscopy**: The number of blood vessels that are visible on a fluoroscopy.
- **Thalassemia**: A blood disorder that can contribute to heart disease.

---

## Why Heart Disease Prediction Matters

Heart disease is a significant global health issue, and predicting its onset early can provide several benefits:
- **Early Detection**: Patients can be advised to adopt healthier lifestyles or receive early treatment to prevent more severe complications.
- **Better Healthcare Resource Allocation**: Hospitals can prioritize patients based on their risk and allocate resources more efficiently.
- **Cost Reduction**: By identifying high-risk individuals early, medical expenses associated with late-stage treatment can be significantly reduced.

Machine learning can play a pivotal role in predicting heart disease because it enables the development of predictive models that can quickly assess a patient’s risk using medical data. These models can help healthcare professionals make better, faster decisions.

---

## Preprocessing

The preprocessing steps involved in this project are crucial for ensuring the dataset is in a format that can be effectively used by the machine learning models.

### Steps Involved:
- **Handling Missing Data**: Missing values in the dataset are handled by imputing values where appropriate or removing rows with missing values if necessary.
Some columns may have missing values, which will be replaced with the mean or median for numerical features or the mode for categorical features.
- **Feature Encoding**: Categorical features, such as Sex, ChestPainType, FastingBloodSugar, etc., are encoded into numerical values using techniques like Label Encoding or One-Hot Encoding.
- **Feature Scaling**: Scaling is done using Min-Max Normalization or Standardization to ensure that all features are on the same scale. This helps improve the performance of distance-based algorithms like KNN and SVM.
- **Splitting Data**: The dataset is split into training and testing sets. Typically, 80% of the data is used for training, and 20% is used for testing.
- **Feature Selection**: Feature selection is the process of selecting the most relevant features for model training. This helps in reducing overfitting, improving model performance, and speeding up training.
- **Correlation Matrix**: Identifying features that are highly correlated with the target variable (Heart Disease), and removing redundant features that don't contribute much to the predictive power.
- **Feature Importance**: Using tree-based models like Random Forest or XGBoost to compute feature importance and select the top features for training the model.

## Models Used

The following machine learning models are used in this project to predict the likelihood of heart disease:

- **Logistic Regression**: A simple and effective model for binary classification.
- **K-Nearest Neighbors (KNN)**: A non-parametric method that makes predictions based on the nearest neighbors in the training data.
- **Support Vector Machine (SVM)**: A classifier that finds the hyperplane that best separates the classes in high-dimensional space.
- **Decision Tree Classifier**: A tree-based model that makes decisions by splitting the data at each node based on the feature that provides the best separation.
- **Random Forest Classifier**: An ensemble method that combines multiple decision trees to improve accuracy and reduce overfitting.
- **Gradient Boosting Classifier**: A boosting method that combines weak models to form a strong classifier.
- **XGBoost**: An optimized implementation of gradient boosting with regularization techniques that improve performance.

---

## Model Training and Evaluation:
Each model is trained using the training data and evaluated on the test data. The models are evaluated using the following metrics:

- **Accuracy**: The percentage of correct predictions made by the model.
- **ROC-AUC**: The area under the ROC curve, which indicates the model's ability to distinguish between the two classes.
- **Precision, Recall, F1-Score**: These metrics are important when dealing with imbalanced datasets.

---

## Interpretation:
- Decision Tree achieved the highest accuracy and AUC, but it may be prone to overfitting given its perfect accuracy. Further tuning could help generalize the model better.
- Random Forest and Gradient Boosting performed excellently and are ideal choices for heart disease prediction.
- Logistic Regression and XGBoost performed well, showing that simpler models like logistic regression can still be powerful when the data is preprocessed effectively.
- KNN and SVM had lower performance, which could be due to hyperparameter choices or the nature of the data.

---

## Future Work

While the models used in this project have shown good performance, there are several ways to improve the heart disease prediction system:

1. **Hyperparameter Tuning**:
   - Experiment with different hyperparameters for the models using techniques like Grid Search or Random Search to identify the optimal settings for better performance.
   
2. **Cross-Validation**:
   - Implement k-fold cross-validation to ensure that the models generalize well across different subsets of the data, reducing the risk of overfitting.
   
3. **Model Interpretability**:
   - Use tools like SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) to make the models more interpretable and explain why the models are making certain predictions.
   
4. **Deep Learning Models**:
   - Implement deep learning models like artificial neural networks (ANNs) to explore if they can provide improved predictive performance over traditional machine learning models.
   
5. **Ensemble Models**:
   - Explore additional ensemble techniques like Stacking or Voting Classifiers to combine the strengths of multiple models and further improve prediction accuracy.

6. **Deploying the Model**:
   - Once the model is optimized, it can be deployed as a web application using frameworks like Flask or Django, allowing real-time heart disease predictions through a user interface.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Conclusion

This project demonstrates how machine learning models can be used to predict heart disease. By preprocessing the data effectively, selecting the right features, and using multiple models such as Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost, we were able to achieve high accuracy and ROC-AUC scores. Among the models tested, **Decision Tree**, **Random Forest**, and **Gradient Boosting** were the top performers.

The project shows the potential of using machine learning for early diagnosis and prediction of heart disease, which can be a valuable tool for healthcare professionals. Further improvements, such as hyperparameter tuning, model interpretability, and exploring deep learning approaches, can help achieve even higher performance and bring this model closer to real-world deployment.

---



