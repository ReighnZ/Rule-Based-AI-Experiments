import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('mental_health_dataset.csv')  # Assuming the CSV file is in the same directory

# Function for manually defining the ground truth (True_Mental_State)
# AI Model's predictions
def determine_mental_state(stress_level, sleep_hours):
    stress_level = stress_level.strip().lower()
   
    if stress_level == 'high' and sleep_hours < 6:
        return 'Not Mentally Healthy'
    elif stress_level == 'moderate' and 6 <= sleep_hours < 8:
        return 'Mentally Healthy'
    elif stress_level == 'low' or sleep_hours >= 8:
        return 'Mentally Healthy'
    else:
        return 'Not Mentally Healthy'  # Default case for any unknown combination

# Manually define the True_Mental_State column based on the rules
df['Mental_State'] = df.apply(lambda row: determine_mental_state(row['Stress_Level'], row['Sleep_Hours']), axis=1)

# Function for determining mental state using the original rule
# Ground truths/correct outcomes
def define_ground_truth(stress_level, sleep_hours):
    stress_level = stress_level.strip().lower()
   
    if stress_level == 'high' and sleep_hours < 6:
        return 'Not Mentally Healthy'
    else:
        return 'Mentally Healthy'

# Apply the mental state rule to each row (predicted mental state)
df['True_Mental_State'] = df.apply(lambda row: define_ground_truth(row['Stress_Level'], row['Sleep_Hours']), axis=1)

# Display the DataFrame
print("Dataset with Defined Mental State:")
print(df[['User_ID', 'Stress_Level', 'Sleep_Hours', 'Mental_State', 'True_Mental_State']])

# Calculate accuracy
accuracy = accuracy_score(df['Mental_State'], df['True_Mental_State'])
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate precision
precision = precision_score(df['Mental_State'], df['True_Mental_State'], pos_label='Mentally Healthy')
print(f"Precision: {precision * 100:.2f}%")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(
    df['Mental_State'].apply(lambda x: 1 if x == 'Mentally Healthy' else 0),
    df['True_Mental_State'].apply(lambda x: 1 if x == 'Mentally Healthy' else 0)
)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % accuracy)
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Display and plot confusion matrix
conf_matrix = confusion_matrix(df['Mental_State'], df['True_Mental_State'], labels=['Not Mentally Healthy', 'Mentally Healthy'])
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Mentally Healthy', 'Mentally Healthy'],
            yticklabels=['Not Mentally Healthy', 'Mentally Healthy'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
