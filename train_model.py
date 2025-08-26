"""
Train EduBoost Health Recommendation Model

This script loads the health check dataset, trains a machine learning model,
and saves it for use with the EduBoost Health Recommendation system.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import joblib
import os


def load_and_preprocess_data(data_path):
    """Load and preprocess the health dataset."""
    print("📊 Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Display basic info about the dataset
    print(f"\n📋 Dataset Overview:")
    print(f"  • Students: {df.shape[0]:,}")
    print(f"  • Features: {df.shape[1]}")
    print(f"  • Mood distribution: {df['Mood'].value_counts().to_dict()}")
    print(f"  • Stress levels: {df['Stress_Level'].value_counts().sort_index().to_dict()}")
    print(f"  • Procrastination levels: {df['Procrastination_Level'].value_counts().sort_index().to_dict()}")
    
    # Select input features and target variables
    input_features = ['Mood', 'Stress_Level', 'Procrastination_Level', 'Current_Sleep_Hours']
    target_variables = [
        'Recommended_Study_Hours', 'Exercise_Minutes_Daily', 'Recommended_Sleep_Hours',
        'Water_Intake_Liters', 'Meditation_Minutes_Daily', 'Screen_Time_Limit_Hours'
    ]
    
    # Extract features and targets
    X = df[input_features].copy()
    y = df[target_variables].copy()
    
    # Rename columns to match our model's expected format
    y.columns = ['study_hours', 'exercise_minutes', 'sleep_hours', 'water_liters', 'meditation_minutes', 'screen_limit']
    
    print(f"\n🎯 Target Statistics:")
    print(y.describe().round(2))
    
    return X, y


def preprocess_features(X):
    """Encode categorical features and scale numerical features."""
    print("\n🔧 Preprocessing features...")
    
    X_processed = X.copy()
    
    # Encode mood using LabelEncoder
    label_encoder = LabelEncoder()
    X_processed['Mood'] = label_encoder.fit_transform(X_processed['Mood'])
    
    print(f"  • Mood encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Scale all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    
    print(f"  • Features scaled using StandardScaler")
    
    return X_scaled, label_encoder, scaler


def train_model(X_train, X_test, y_train, y_test):
    """Train the multi-output regression model."""
    print("\n🚀 Training model...")
    
    # Use RandomForestRegressor as the base estimator for multi-output regression
    base_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Wrap in MultiOutputRegressor for multiple target prediction
    model = MultiOutputRegressor(base_model)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics for each target
    print(f"\n📈 Model Performance:")
    target_names = ['study_hours', 'exercise_minutes', 'sleep_hours', 'water_liters', 'meditation_minutes', 'screen_limit']
    
    overall_r2_scores = []
    overall_rmse_scores = []
    
    for i, target in enumerate(target_names):
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        overall_r2_scores.append(r2)
        overall_rmse_scores.append(rmse)
        print(f"  • {target:<20}: R² = {r2:.3f}, RMSE = {rmse:.3f}")
    
    # Overall performance
    avg_r2 = np.mean(overall_r2_scores)
    avg_rmse = np.mean(overall_rmse_scores)
    
    print(f"\n🎯 Overall Performance:")
    print(f"  • Average R² Score: {avg_r2:.3f}")
    print(f"  • Average RMSE: {avg_rmse:.3f}")
    
    return model


def save_model_and_preprocessors(model, label_encoder, scaler, model_path):
    """Save the trained model and preprocessors."""
    print(f"\n💾 Saving model to {model_path}...")
    
    # Create a dictionary containing all components
    model_package = {
        'model': model,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'feature_names': ['Mood', 'Stress_Level', 'Procrastination_Level', 'Current_Sleep_Hours'],
        'target_names': ['study_hours', 'exercise_minutes', 'sleep_hours', 'water_liters', 'meditation_minutes', 'screen_limit'],
        'model_info': {
            'model_type': 'MultiOutputRegressor with RandomForest',
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sklearn_version': '1.0+'
        }
    }
    
    # Save using joblib
    joblib.dump(model_package, model_path)
    print(f"✓ Model saved successfully!")
    
    # Verify the saved file
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
    print(f"  • File size: {file_size:.2f} MB")


def test_saved_model(model_path):
    """Test the saved model with sample predictions."""
    print(f"\n🧪 Testing saved model...")
    
    # Load the saved model
    model_package = joblib.load(model_path)
    model = model_package['model']
    label_encoder = model_package['label_encoder']
    scaler = model_package['scaler']
    
    # Test samples
    test_cases = [
        {'mood': 'Happy', 'stress': 2, 'procrastination': 2, 'sleep': 8.0},
        {'mood': 'Stressed', 'stress': 4, 'procrastination': 4, 'sleep': 6.0},
        {'mood': 'Neutral', 'stress': 3, 'procrastination': 3, 'sleep': 7.5},
        {'mood': 'Sad', 'stress': 4, 'procrastination': 5, 'sleep': 5.5}
    ]
    
    print(f"  Test predictions:")
    for i, case in enumerate(test_cases, 1):
        # Encode mood
        mood_encoded = label_encoder.transform([case['mood']])[0]
        
        # Create feature array
        features = np.array([[mood_encoded, case['stress'], case['procrastination'], case['sleep']]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        print(f"    {i}. {case['mood']}, Stress:{case['stress']}, Proc:{case['procrastination']}, Sleep:{case['sleep']}h")
        print(f"       → Study:{prediction[0]:.1f}h, Exercise:{prediction[1]:.0f}min, Sleep:{prediction[2]:.1f}h")
        print(f"         Water:{prediction[3]:.1f}L, Meditation:{prediction[4]:.0f}min, Screen:{prediction[5]:.1f}h")


def main():
    """Main training pipeline."""
    print("🎓 EduBoost Health Model Training Pipeline")
    print("=" * 60)
    
    # Paths
    data_path = ".github/health_model/health_check_100k_dataset.csv"
    model_path = "eduboost_health_recommendation_model.pkl"
    
    try:
        # Load and preprocess data
        X, y = load_and_preprocess_data(data_path)
        
        # Preprocess features
        X_processed, label_encoder, scaler = preprocess_features(X)
        
        # Split data
        print(f"\n✂️  Splitting data (80% train, 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        print(f"  • Training set: {X_train.shape[0]} samples")
        print(f"  • Test set: {X_test.shape[0]} samples")
        
        # Train model
        model = train_model(X_train, X_test, y_train, y_test)
        
        # Save model and preprocessors
        save_model_and_preprocessors(model, label_encoder, scaler, model_path)
        
        # Test the saved model
        test_saved_model(model_path)
        
        print(f"\n" + "=" * 60)
        print("✅ Training completed successfully!")
        print(f"📁 Model saved as: {model_path}")
        print(f"🚀 Ready to use with EduBoostHealthModel class!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
