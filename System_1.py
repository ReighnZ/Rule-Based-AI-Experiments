import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
# Load the main dataset
df = pd.read_csv('mental_health_dataset.csv')

# Define the recommendation function based on mental health rules
# AI Model's predictions
def mental_health_recommendation(row):
    condition = row['Mental_Health_Condition']
    severity = row['Severity']
    stress_level = row['Stress_Level']
    sleep_hours = row['Sleep_Hours']
   
    # Rule 1: High Severity
    if condition == 'Yes' and severity == 'High':
        return "Seek immediate professional help."
   
    # Rule 2: Medium Severity
    elif condition == 'Yes' and severity == 'Medium':
        if stress_level in ['High', 'Medium']:
            return "Consider professional consultation."
        elif stress_level == 'Low':
            return "Monitor condition and consider follow-up consultation."
   
    # Rule 3: Low Severity
    elif condition == 'Yes' and severity == 'Low':
        if stress_level == 'High' and sleep_hours < 6:
            return "Seek professional help due to stress and poor sleep."
        elif stress_level == 'Medium' and 6 <= sleep_hours <= 8:
            return "Try self-care strategies like exercise and relaxation."
        elif stress_level == 'Low' and sleep_hours > 8:
            return "Maintain self-care and regular check-ups."
   
    # Rule 4: No Mental Health Condition
    elif condition == 'No':
        if stress_level in ['High', 'Medium']:
            return "Monitor stress and adopt stress-relieving techniques."
        if stress_level == 'Low':
            return "Maintain wellness practices to preserve mental health."
        if sleep_hours < 6 and stress_level == 'High':
            return "Consider seeking professional consultation for stress and sleep issues."


    return "No action needed."


# Apply the recommendation function to each row in the DataFrame
df['Recommendation'] = df.apply(mental_health_recommendation, axis=1)


# Display the DataFrame with recommendations
print("Dataset with Recommendations:")
print(df)


# Test data for accuracy and ROC curve
# Ground truths/correct outcomes
test_data = [
    {'User_ID': 1, 'Mental_Health_Condition': 'Yes', 'Severity': 'High', 'Stress_Level': 'Medium', 'Sleep_Hours': 7, 'Expected_Recommendation': 'Seek immediate professional help.'},
    {'User_ID': 2, 'Mental_Health_Condition': 'Yes', 'Severity': 'Medium', 'Stress_Level': 'High', 'Sleep_Hours': 5, 'Expected_Recommendation': 'Consider professional consultation.'},
    {'User_ID': 3, 'Mental_Health_Condition': 'Yes', 'Severity': 'Low', 'Stress_Level': 'Low', 'Sleep_Hours': 9, 'Expected_Recommendation': 'Maintain self-care and regular check-ups.'},
    {'User_ID': 4, 'Mental_Health_Condition': 'No', 'Severity': 'N/A', 'Stress_Level': 'High', 'Sleep_Hours': 5, 'Expected_Recommendation': 'Consider seeking professional consultation for stress and sleep issues.'},
]


# Convert test data to DataFrame
test_df = pd.DataFrame(test_data)


# Apply the recommendation function to the test data
test_df['Recommendation'] = test_df.apply(mental_health_recommendation, axis=1)


# Calculate accuracy
accuracy = accuracy_score(test_df['Expected_Recommendation'], test_df['Recommendation'])
print(f"\nAccuracy: {accuracy:.2f}")

# Calculate precision for each class, with zero_division=0 to handle undefined cases
precision = precision_score(test_df['Expected_Recommendation'], test_df['Recommendation'], average=None, labels=test_df['Expected_Recommendation'].unique(), zero_division=0)

# Print precision for each class
print("\nPrecision for each class:")
for rec, prec in zip(test_df['Expected_Recommendation'].unique(), precision):
    print(f"{rec}: {prec:.2f}")


# Confusion matrix
conf_matrix = confusion_matrix(test_df['Expected_Recommendation'], test_df['Recommendation'], labels=test_df['Expected_Recommendation'].unique())


# Plot confusion matrix using Seaborn heatmap for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=test_df['Expected_Recommendation'].unique(), yticklabels=test_df['Expected_Recommendation'].unique())
plt.xlabel('Predicted Recommendation')
plt.ylabel('Expected Recommendation')
plt.title('Confusion Matrix')
plt.show()


# Binarize the Expected and Predicted recommendations for ROC curve
all_recommendations = test_df['Expected_Recommendation'].unique()
y_true = label_binarize(test_df['Expected_Recommendation'], classes=all_recommendations)
y_pred = label_binarize(test_df['Recommendation'], classes=all_recommendations)


# Plot ROC curve for each class (one-vs-rest)
plt.figure(figsize=(10, 8))
for i, recommendation in enumerate(all_recommendations):
    fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve for {recommendation} (AUC = {roc_auc:.2f})')


# Plot settings for ROC curve
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - One-vs-Rest')
plt.legend(loc='lower right')
plt.show()