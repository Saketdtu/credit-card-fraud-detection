# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading credit card fraud dataset...")
try:
    df = pd.read_csv('creditcard_2023.csv')
    print(f"Dataset loaded successfully! Shape: {df.shape}")
except FileNotFoundError:
    print("Dataset file not found. Creating sample data...")
    # Create sample fraud detection data
    np.random.seed(42)
    n_samples = 10000
    n_fraud = 100
    
    # Generate normal transactions
    normal_data = {
        'Amount': np.random.exponential(50, n_samples - n_fraud),
        'Time': np.random.randint(0, 86400, n_samples - n_fraud),
        'V1': np.random.normal(0, 1, n_samples - n_fraud),
        'V2': np.random.normal(0, 1, n_samples - n_fraud),
        'V3': np.random.normal(0, 1, n_samples - n_fraud),
        'V4': np.random.normal(0, 1, n_samples - n_fraud),
        'Class': [0] * (n_samples - n_fraud)
    }
    
    # Generate fraud transactions (different distribution)
    fraud_data = {
        'Amount': np.random.exponential(200, n_fraud),
        'Time': np.random.randint(0, 86400, n_fraud),
        'V1': np.random.normal(2, 1.5, n_fraud),
        'V2': np.random.normal(-1, 1.2, n_fraud),
        'V3': np.random.normal(1.5, 1.8, n_fraud),
        'V4': np.random.normal(-0.5, 1.3, n_fraud),
        'Class': [1] * n_fraud
    }
    
    # Combine data
    all_data = {}
    for key in normal_data.keys():
        all_data[key] = normal_data[key] + fraud_data[key]
    
    df = pd.DataFrame(all_data)
    print(f"Sample dataset created! Shape: {df.shape}")

# Display basic information
print("\nDataset Information:")
print(df.info())

print("\nFirst 5 rows:")
print(df.head())

print("\nClass distribution:")
print(df['Class'].value_counts())

print(f"\nFraud percentage: {(df['Class'].sum() / len(df)) * 100:.2f}%")

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# Exploratory Data Analysis
plt.figure(figsize=(15, 12))

# Plot 1: Class distribution
plt.subplot(3, 3, 1)
class_counts = df['Class'].value_counts()
plt.pie(class_counts.values, labels=['Normal', 'Fraud'], autopct='%1.1f%%')
plt.title('Transaction Class Distribution')

# Plot 2: Amount distribution
plt.subplot(3, 3, 2)
plt.hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.5, label='Normal', density=True)
plt.hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.5, label='Fraud', density=True)
plt.legend()
plt.xlabel('Amount')
plt.ylabel('Density')
plt.title('Amount Distribution by Class')
plt.yscale('log')

# Plot 3: Time distribution
if 'Time' in df.columns:
    plt.subplot(3, 3, 3)
    plt.hist(df[df['Class']==0]['Time'], bins=50, alpha=0.5, label='Normal', density=True)
    plt.hist(df[df['Class']==1]['Time'], bins=50, alpha=0.5, label='Fraud', density=True)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Density')
    plt.title('Time Distribution by Class')

# Plot 4: Correlation matrix
plt.subplot(3, 3, 4)
# Select numeric columns for correlation
numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]  # First 10 numeric columns
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')

# Plot 5: Amount vs Class boxplot
plt.subplot(3, 3, 5)
df.boxplot(column='Amount', by='Class', ax=plt.gca())
plt.title('Amount Distribution by Class')
plt.suptitle('')  # Remove default title

# Plot 6: Feature distribution (V1-V4 if available)
feature_cols = [col for col in df.columns if col.startswith('V')][:4]
if feature_cols:
    for i, col in enumerate(feature_cols):
        plt.subplot(3, 3, 6+i)
        plt.hist(df[df['Class']==0][col], bins=30, alpha=0.5, label='Normal', density=True)
        plt.hist(df[df['Class']==1][col], bins=30, alpha=0.5, label='Fraud', density=True)
        plt.legend()
        plt.title(f'{col} Distribution')

plt.tight_layout()
plt.savefig('fraud_detection_eda.png', dpi=300, bbox_inches='tight')
plt.show()

# Data preprocessing
print("\nPreprocessing data...")

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Check feature types
print(f"\nFeature columns: {X.columns.tolist()}")

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Features scaled successfully!")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Fraud in train set: {y_train.sum()} ({(y_train.sum()/len(y_train))*100:.2f}%)")
print(f"Fraud in test set: {y_test.sum()} ({(y_test.sum()/len(y_test))*100:.2f}%)")

# Train models with imbalanced data
print("\n" + "="*50)
print("TRAINING MODELS WITH IMBALANCED DATA")
print("="*50)

# Logistic Regression
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_prob = lr_model.predict_proba(X_test)[:, 1]

# Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# Handle class imbalance with SMOTE
print("\n" + "="*50)
print("HANDLING CLASS IMBALANCE WITH SMOTE")
print("="*50)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"After SMOTE - Train set size: {X_train_smote.shape[0]}")
print(f"After SMOTE - Fraud cases: {y_train_smote.sum()} ({(y_train_smote.sum()/len(y_train_smote))*100:.2f}%)")

# Train models with balanced data
print("\nTraining models with balanced data...")

# Logistic Regression with SMOTE
lr_smote = LogisticRegression(random_state=42)
lr_smote.fit(X_train_smote, y_train_smote)
lr_smote_pred = lr_smote.predict(X_test)
lr_smote_prob = lr_smote.predict_proba(X_test)[:, 1]

# Random Forest with SMOTE
rf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)
rf_smote_pred = rf_smote.predict(X_test)
rf_smote_prob = rf_smote.predict_proba(X_test)[:, 1]

# Evaluation function
def evaluate_model(y_true, y_pred, y_prob, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'AUC_ROC': auc
    }

# Evaluate all models
print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)

results = []
results.append(evaluate_model(y_test, lr_pred, lr_prob, "Logistic Regression"))
results.append(evaluate_model(y_test, rf_pred, rf_prob, "Random Forest"))
results.append(evaluate_model(y_test, lr_smote_pred, lr_smote_prob, "Logistic Regression + SMOTE"))
results.append(evaluate_model(y_test, rf_smote_pred, rf_smote_prob, "Random Forest + SMOTE"))

# Create results DataFrame
results_df = pd.DataFrame(results)
print(f"\nSummary of Results:")
print(results_df.round(4))

# Visualization of results
plt.figure(figsize=(15, 12))

# Plot 1: Model comparison
plt.subplot(3, 3, 1)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC']
x = np.arange(len(results))
width = 0.15

for i, metric in enumerate(metrics):
    plt.bar(x + i*width, results_df[metric], width, label=metric)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x + width*2, results_df['Model'], rotation=45)
plt.legend()

# Plot 2: Confusion matrices
models_data = [
    (rf_smote_pred, "Random Forest + SMOTE"),
    (lr_smote_pred, "Logistic Regression + SMOTE")
]

for i, (pred, name) in enumerate(models_data):
    plt.subplot(3, 3, 2+i)
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

# Plot 4: ROC Curves
plt.subplot(3, 3, 4)
models_prob = [
    (rf_prob, "Random Forest"),
    (rf_smote_prob, "Random Forest + SMOTE"),
    (lr_prob, "Logistic Regression"),
    (lr_smote_prob, "Logistic Regression + SMOTE")
]

for prob, name in models_prob:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()

# Plot 5: Feature importance (Random Forest)
plt.subplot(3, 3, 5)
if hasattr(rf_smote, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_smote.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.title('Top 10 Feature Importance (RF + SMOTE)')
    plt.xlabel('Importance')

# Plot 6: Prediction probability distribution
plt.subplot(3, 3, 6)
plt.hist(rf_smote_prob[y_test==0], bins=50, alpha=0.5, label='Normal', density=True)
plt.hist(rf_smote_prob[y_test==1], bins=50, alpha=0.5, label='Fraud', density=True)
plt.xlabel('Prediction Probability')
plt.ylabel('Density')
plt.title('Prediction Probability Distribution')
plt.legend()

plt.tight_layout()
plt.savefig('fraud_detection_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Detailed classification report for best model
best_model_idx = results_df['F1_Score'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
print(f"\nBest Model: {best_model_name}")

if best_model_name == "Random Forest + SMOTE":
    best_pred = rf_smote_pred
elif best_model_name == "Logistic Regression + SMOTE":
    best_pred = lr_smote_pred
elif best_model_name == "Random Forest":
    best_pred = rf_pred
else:
    best_pred = lr_pred

print(f"\nDetailed Classification Report for {best_model_name}:")
print(classification_report(y_test, best_pred))

# Sample fraud detection function
def detect_fraud(transaction_features, model=rf_smote, scaler=scaler, threshold=0.5):
    """
    Detect if a transaction is fraudulent
    
    Parameters:
    transaction_features: array-like, features of the transaction
    model: trained model
    scaler: fitted scaler
    threshold: probability threshold for classification
    
    Returns:
    dict with prediction and probability
    """
    # Scale features
    scaled_features = scaler.transform([transaction_features])
    
    # Get prediction and probability
    probability = model.predict_proba(scaled_features)[0, 1]
    prediction = 1 if probability >= threshold else 0
    
    return {
        'is_fraud': bool(prediction),
        'fraud_probability': probability,
        'risk_level': 'High' if probability >= 0.7 else 'Medium' if probability >= 0.3 else 'Low'
    }

# Test with sample transactions
print(f"\nSample Fraud Detection:")
print("-" * 50)

# Create sample transactions
sample_transactions = X_test[:5]
actual_labels = y_test.iloc[:5]

for i, (transaction, actual) in enumerate(zip(sample_transactions, actual_labels)):
    result = detect_fraud(transaction)
    print(f"Transaction {i+1}:")
    print(f"  Actual: {'Fraud' if actual else 'Normal'}")
    print(f"  Predicted: {'Fraud' if result['is_fraud'] else 'Normal'}")
    print(f"  Fraud Probability: {result['fraud_probability']:.3f}")
    print(f"  Risk Level: {result['risk_level']}")
    print("-" * 30)

# Final summary
summary = {
    'Total_Transactions': len(df),
    'Fraud_Transactions': df['Class'].sum(),
    'Fraud_Rate': (df['Class'].sum() / len(df)) * 100,
    'Best_Model': best_model_name,
    'Best_F1_Score': results_df.loc[best_model_idx, 'F1_Score'],
    'Best_Precision': results_df.loc[best_model_idx, 'Precision'],
    'Best_Recall': results_df.loc[best_model_idx, 'Recall']
}

print(f"\nProject Summary:")
print("="*50)
for key, value in summary.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

print("\nProject completed successfully!")
print("Generated files:")
print("- fraud_detection_eda.png")
print("- fraud_detection_results.png")
