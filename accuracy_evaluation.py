"""
EduBoost Health Model - Accuracy and Performance Evaluation

This script provides detailed accuracy metrics, performance analysis,
and model evaluation for the EduBoost Health Recommendation Model.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from eduboost_health_model import EduBoostHealthModel


def load_test_data():
    """Load and prepare test data for evaluation."""
    print("📊 Loading test dataset...")
    
    # Load the original dataset
    df = pd.read_csv(".github/health_model/health_check_100k_dataset.csv")
    
    # Prepare features and targets
    input_features = ['Mood', 'Stress_Level', 'Procrastination_Level', 'Current_Sleep_Hours']
    target_variables = [
        'Recommended_Study_Hours', 'Exercise_Minutes_Daily', 'Recommended_Sleep_Hours',
        'Water_Intake_Liters', 'Meditation_Minutes_Daily', 'Screen_Time_Limit_Hours'
    ]
    
    X = df[input_features].copy()
    y = df[target_variables].copy()
    y.columns = ['study_hours', 'exercise_minutes', 'sleep_hours', 'water_liters', 'meditation_minutes', 'screen_limit']
    
    return X, y


def evaluate_model_accuracy():
    """Evaluate the model's accuracy using various metrics."""
    print("🎯 EduBoost Health Model - Accuracy Evaluation")
    print("=" * 60)
    
    # Load model
    try:
        model_package = joblib.load("eduboost_health_recommendation_model.pkl")
        trained_model = model_package['model']
        label_encoder = model_package['label_encoder']
        scaler = model_package['scaler']
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load and prepare data
    X, y = load_test_data()
    
    # Preprocess features (same as training)
    X_processed = X.copy()
    X_processed['Mood'] = label_encoder.transform(X_processed['Mood'])
    X_scaled = scaler.transform(X_processed)
    
    # Split data (using same random state as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📋 Dataset Information:")
    print(f"  • Total samples: {len(X):,}")
    print(f"  • Training samples: {len(X_train):,}")
    print(f"  • Test samples: {len(X_test):,}")
    
    # Make predictions
    print(f"\n🔮 Making predictions on test set...")
    y_pred = trained_model.predict(X_test)
    
    # Calculate detailed metrics for each target
    target_names = ['study_hours', 'exercise_minutes', 'sleep_hours', 'water_liters', 'meditation_minutes', 'screen_limit']
    target_units = ['hours', 'minutes', 'hours', 'liters', 'minutes', 'hours']
    
    print(f"\n📈 DETAILED ACCURACY METRICS")
    print("=" * 80)
    print(f"{'Target':<20} {'R² Score':<12} {'RMSE':<12} {'MAE':<12} {'MAPE':<12} {'Accuracy':<12}")
    print("-" * 80)
    
    overall_metrics = {
        'r2_scores': [],
        'rmse_scores': [],
        'mae_scores': [],
        'mape_scores': [],
        'accuracy_scores': []
    }
    
    for i, (target, unit) in enumerate(zip(target_names, target_units)):
        y_true = y_test.iloc[:, i]
        y_pred_target = y_pred[:, i]
        
        # Calculate metrics
        r2 = r2_score(y_true, y_pred_target)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_target))
        mae = mean_absolute_error(y_true, y_pred_target)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred_target) / y_true)) * 100
        
        # Calculate accuracy as percentage (100% - MAPE)
        accuracy = max(0, 100 - mape)
        
        # Store metrics
        overall_metrics['r2_scores'].append(r2)
        overall_metrics['rmse_scores'].append(rmse)
        overall_metrics['mae_scores'].append(mae)
        overall_metrics['mape_scores'].append(mape)
        overall_metrics['accuracy_scores'].append(accuracy)
        
        print(f"{target:<20} {r2:<12.3f} {rmse:<12.3f} {mae:<12.3f} {mape:<12.1f}% {accuracy:<12.1f}%")
    
    # Overall performance summary
    print("\n" + "=" * 80)
    print("🎯 OVERALL MODEL PERFORMANCE")
    print("=" * 80)
    
    avg_r2 = np.mean(overall_metrics['r2_scores'])
    avg_rmse = np.mean(overall_metrics['rmse_scores'])
    avg_mae = np.mean(overall_metrics['mae_scores'])
    avg_mape = np.mean(overall_metrics['mape_scores'])
    avg_accuracy = np.mean(overall_metrics['accuracy_scores'])
    
    print(f"Average R² Score:          {avg_r2:.3f} ({avg_r2*100:.1f}%)")
    print(f"Average RMSE:              {avg_rmse:.3f}")
    print(f"Average MAE:               {avg_mae:.3f}")
    print(f"Average MAPE:              {avg_mape:.1f}%")
    print(f"Average Accuracy:          {avg_accuracy:.1f}%")
    
    # Performance interpretation
    print(f"\n🏆 PERFORMANCE INTERPRETATION")
    print("-" * 40)
    
    if avg_accuracy >= 90:
        performance_grade = "Excellent (A+)"
        interpretation = "Outstanding predictive performance"
    elif avg_accuracy >= 85:
        performance_grade = "Very Good (A)"
        interpretation = "Very strong predictive performance"
    elif avg_accuracy >= 80:
        performance_grade = "Good (B+)"
        interpretation = "Good predictive performance"
    elif avg_accuracy >= 75:
        performance_grade = "Satisfactory (B)"
        interpretation = "Acceptable predictive performance"
    elif avg_accuracy >= 70:
        performance_grade = "Fair (C)"
        interpretation = "Fair predictive performance"
    else:
        performance_grade = "Needs Improvement"
        interpretation = "Requires model optimization"
    
    print(f"Overall Grade:             {performance_grade}")
    print(f"Interpretation:            {interpretation}")
    
    # Best and worst performing targets
    best_target_idx = np.argmax(overall_metrics['accuracy_scores'])
    worst_target_idx = np.argmin(overall_metrics['accuracy_scores'])
    
    print(f"\n🥇 Best Performing Target:   {target_names[best_target_idx]} ({overall_metrics['accuracy_scores'][best_target_idx]:.1f}% accuracy)")
    print(f"🔄 Needs Improvement:       {target_names[worst_target_idx]} ({overall_metrics['accuracy_scores'][worst_target_idx]:.1f}% accuracy)")
    
    return overall_metrics, target_names


def test_real_world_accuracy():
    """Test accuracy with the EduBoost model class using real examples."""
    print(f"\n🧪 REAL-WORLD ACCURACY TEST")
    print("=" * 60)
    
    # Initialize the model
    model = EduBoostHealthModel(verbose=False)
    
    # Test cases with known expected ranges
    test_cases = [
        {
            'input': {'mood': 'Happy', 'stress_level': 1, 'procrastination_level': 1, 'sleep_hours': 8.5},
            'expected_ranges': {
                'study_hours': (22, 28),
                'exercise_minutes': (30, 50),
                'sleep_hours': (7.5, 8.5),
                'water_liters': (2.0, 3.0),
                'meditation_minutes': (10, 20),
                'screen_limit': (1.0, 2.0)
            }
        },
        {
            'input': {'mood': 'Stressed', 'stress_level': 5, 'procrastination_level': 5, 'sleep_hours': 5.0},
            'expected_ranges': {
                'study_hours': (12, 18),
                'exercise_minutes': (50, 60),
                'sleep_hours': (8.5, 9.5),
                'water_liters': (3.0, 4.0),
                'meditation_minutes': (25, 40),
                'screen_limit': (2.5, 4.0)
            }
        },
        {
            'input': {'mood': 'Neutral', 'stress_level': 3, 'procrastination_level': 3, 'sleep_hours': 7.5},
            'expected_ranges': {
                'study_hours': (18, 24),
                'exercise_minutes': (40, 55),
                'sleep_hours': (7.8, 8.5),
                'water_liters': (2.5, 3.2),
                'meditation_minutes': (18, 25),
                'screen_limit': (1.5, 2.5)
            }
        }
    ]
    
    total_predictions = 0
    accurate_predictions = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['input']['mood']}, Stress: {test_case['input']['stress_level']}")
        
        try:
            result = model.predict(**test_case['input'])
            predictions = result['detailed_metrics']
            
            case_accurate = 0
            case_total = 0
            
            for metric, predicted_value in predictions.items():
                if metric in test_case['expected_ranges']:
                    min_val, max_val = test_case['expected_ranges'][metric]
                    is_accurate = min_val <= predicted_value <= max_val
                    
                    status = "✓" if is_accurate else "✗"
                    print(f"  {metric}: {predicted_value} {status} (expected: {min_val}-{max_val})")
                    
                    if is_accurate:
                        case_accurate += 1
                        accurate_predictions += 1
                    case_total += 1
                    total_predictions += 1
            
            case_accuracy = (case_accurate / case_total) * 100
            print(f"  Case Accuracy: {case_accuracy:.1f}% ({case_accurate}/{case_total})")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    overall_accuracy = (accurate_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"\n🎯 Real-World Test Accuracy: {overall_accuracy:.1f}% ({accurate_predictions}/{total_predictions})")
    
    return overall_accuracy


def display_confidence_analysis():
    """Analyze model confidence scores."""
    print(f"\n🔍 MODEL CONFIDENCE ANALYSIS")
    print("=" * 60)
    
    model = EduBoostHealthModel(verbose=False)
    
    # Test various input combinations to analyze confidence patterns
    test_inputs = []
    confidences = []
    
    moods = ['Happy', 'Neutral', 'Stressed', 'Sad']
    stress_levels = [1, 2, 3, 4, 5]
    procrastination_levels = [1, 2, 3, 4, 5]
    sleep_hours = [6.0, 7.0, 8.0, 9.0]
    
    print("Analyzing confidence patterns across different input combinations...")
    
    sample_count = 0
    for mood in moods:
        for stress in stress_levels[:3]:  # Sample subset
            for proc in procrastination_levels[:3]:  # Sample subset
                for sleep in sleep_hours[:2]:  # Sample subset
                    try:
                        result = model.predict(mood, stress, proc, sleep)
                        test_inputs.append(f"{mood}-S{stress}-P{proc}-{sleep}h")
                        confidences.append(result['model_confidence'])
                        sample_count += 1
                        if sample_count >= 20:  # Limit sample size
                            break
                    except Exception:
                        continue  # Skip failed predictions
                if sample_count >= 20:
                    break
            if sample_count >= 20:
                break
        if sample_count >= 20:
            break
    
    if confidences:
        avg_confidence = np.mean(confidences)
        min_confidence = np.min(confidences)
        max_confidence = np.max(confidences)
        std_confidence = np.std(confidences)
        
        print(f"\nConfidence Statistics (from {len(confidences)} predictions):")
        print(f"  • Average Confidence: {avg_confidence:.3f} ({avg_confidence*100:.1f}%)")
        print(f"  • Minimum Confidence: {min_confidence:.3f} ({min_confidence*100:.1f}%)")
        print(f"  • Maximum Confidence: {max_confidence:.3f} ({max_confidence*100:.1f}%)")
        print(f"  • Standard Deviation: {std_confidence:.3f}")
        
        # Confidence interpretation
        if avg_confidence >= 0.9:
            conf_grade = "Very High"
        elif avg_confidence >= 0.8:
            conf_grade = "High"
        elif avg_confidence >= 0.7:
            conf_grade = "Good"
        elif avg_confidence >= 0.6:
            conf_grade = "Moderate"
        else:
            conf_grade = "Low"
        
        print(f"  • Confidence Level: {conf_grade}")


def main():
    """Main evaluation function."""
    print("🎓 EduBoost Health Model - Comprehensive Accuracy Report")
    print("=" * 70)
    
    try:
        # 1. Detailed model accuracy evaluation
        metrics, targets = evaluate_model_accuracy()
        
        # 2. Real-world accuracy testing
        real_world_accuracy = test_real_world_accuracy()
        
        # 3. Confidence analysis
        display_confidence_analysis()
        
        # 4. Final summary
        print(f"\n" + "=" * 70)
        print("📋 FINAL ACCURACY SUMMARY")
        print("=" * 70)
        
        avg_accuracy = np.mean(metrics['accuracy_scores'])
        print(f"Statistical Model Accuracy:    {avg_accuracy:.1f}%")
        print(f"Real-World Test Accuracy:      {real_world_accuracy:.1f}%")
        print(f"Average R² Score:              {np.mean(metrics['r2_scores']):.3f}")
        print(f"Model Status:                  {'✅ Production Ready' if avg_accuracy >= 75 else '⚠️ Needs Improvement'}")
        
        print(f"\n🎯 The EduBoost Health Model demonstrates {avg_accuracy:.1f}% accuracy")
        print(f"   in predicting personalized health recommendations for students.")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


if __name__ == "__main__":
    main()
