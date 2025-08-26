"""
Example usage of the EduBoost Health Recommendation Model

This script demonstrates how to use the EduBoost Health Model for both
single student predictions and batch processing.
"""

from eduboost_health_model import EduBoostHealthModel


def main():
    """Main function demonstrating model usage."""
    
    print("🎓 EduBoost Health Recommendation Model - Example Usage")
    print("=" * 60)
    
    # Initialize the model with verbose output
    try:
        model = EduBoostHealthModel(
            model_path='eduboost_health_recommendation_model.pkl',
            verbose=True
        )
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"⚠️  Model file not found, using heuristic predictions: {e}")
        model = EduBoostHealthModel(verbose=True)
    
    # Example 1: Single student prediction
    print("\n" + "="*60)
    print("📝 EXAMPLE 1: Single Student Prediction")
    print("="*60)
    
    try:
        result = model.predict(
            mood='Stressed',
            stress_level=4,
            procrastination_level=3,
            sleep_hours=6.5
        )
        
        print("\n📊 PREDICTION RESULTS:")
        print("-" * 30)
        print(f"Input Analysis:")
        print(f"  • Mood: {result['input_analysis']['mood']}")
        print(f"  • Stress Level: {result['input_analysis']['stress_level']} ({result['input_analysis']['stress_category']})")
        print(f"  • Procrastination: {result['input_analysis']['procrastination_level']}")
        print(f"  • Sleep: {result['input_analysis']['sleep_hours']}h ({result['input_analysis']['sleep_quality']})")
        
        print(f"\nRecommended Metrics:")
        metrics = result['detailed_metrics']
        print(f"  • Study Hours: {metrics['study_hours']}h")
        print(f"  • Exercise: {metrics['exercise_minutes']} minutes")
        print(f"  • Sleep: {metrics['sleep_hours']} hours")
        print(f"  • Water: {metrics['water_liters']} liters")
        print(f"  • Meditation: {metrics['meditation_minutes']} minutes")
        print(f"  • Screen Limit: {metrics['screen_limit']} hours")
        
        print(f"\nModel Confidence: {result['model_confidence']:.1%}")
        
        print(f"\n📚 PERSONALIZED PLANS:")
        print("-" * 30)
        plans = result['recommendations']
        print(f"\n📖 Study Plan:")
        print(f"   {plans['study_plan']}")
        print(f"\n💪 Physical Plan:")
        print(f"   {plans['physical_plan']}")
        print(f"\n🧘 Emotional Plan:")
        print(f"   {plans['emotional_plan']}")
        
    except Exception as e:
        print(f"❌ Error in single prediction: {e}")
    
    # Example 2: Batch predictions
    print("\n" + "="*60)
    print("👥 EXAMPLE 2: Batch Student Predictions")
    print("="*60)
    
    # Sample student data
    students_data = [
        {
            'mood': 'Happy',
            'stress_level': 2,
            'procrastination_level': 2,
            'sleep_hours': 8.0
        },
        {
            'mood': 'Stressed',
            'stress_level': 5,
            'procrastination_level': 4,
            'sleep_hours': 5.5
        },
        {
            'mood': 'Neutral',
            'stress_level': 3,
            'procrastination_level': 3,
            'sleep_hours': 7.5
        },
        {
            'mood': 'Sad',
            'stress_level': 4,
            'procrastination_level': 5,
            'sleep_hours': 6.0
        }
    ]
    
    try:
        batch_results = model.batch_predict(students_data)
        
        print(f"\n📊 BATCH RESULTS SUMMARY:")
        print("-" * 50)
        print(f"{'Student':<10} {'Mood':<10} {'Stress':<8} {'Study':<8} {'Exercise':<10} {'Confidence':<12}")
        print("-" * 50)
        
        for result in batch_results:
            if 'error' not in result:
                student_id = result['student_id']
                mood = result['input_analysis']['mood']
                stress = result['input_analysis']['stress_level']
                study = f"{result['detailed_metrics']['study_hours']}h"
                exercise = f"{result['detailed_metrics']['exercise_minutes']}min"
                confidence = f"{result['model_confidence']:.1%}"
                
                print(f"{student_id:<10} {mood:<10} {stress:<8} {study:<8} {exercise:<10} {confidence:<12}")
            else:
                print(f"Student {result['student_id']}: Error - {result['error']}")
        
    except Exception as e:
        print(f"❌ Error in batch predictions: {e}")
    
    # Example 3: Input validation
    print("\n" + "="*60)
    print("🔍 EXAMPLE 3: Input Validation")
    print("="*60)
    
    # Test invalid inputs
    invalid_cases = [
        ("Invalid mood", "InvalidMood", 3, 3, 7.0),
        ("Stress level too high", "Happy", 6, 3, 7.0),
        ("Negative sleep", "Happy", 3, 3, -1.0),
        ("Non-numeric stress", "Happy", "high", 3, 7.0)
    ]
    
    for case_name, mood, stress, procrastination, sleep in invalid_cases:
        try:
            result = model.predict(mood, stress, procrastination, sleep)
            print(f"✓ {case_name}: Prediction succeeded (unexpected)")
        except Exception as e:
            print(f"❌ {case_name}: {str(e)}")
    
    # Example 4: Model information
    print("\n" + "="*60)
    print("🔧 EXAMPLE 4: Model Information")
    print("="*60)
    
    model_info = model.get_model_info()
    print(f"\nModel Details:")
    print(f"  • Model Type: {model_info['model_type']}")
    print(f"  • Model Path: {model_info['model_path']}")
    print(f"  • Valid Moods: {', '.join(model_info['valid_moods'])}")
    print(f"  • Target Columns: {', '.join(model_info['target_columns'])}")
    print(f"  • Verbose Mode: {model_info['verbose_mode']}")
    
    print(f"\nPrediction Bounds:")
    for target, (min_val, max_val) in model_info['prediction_bounds'].items():
        print(f"  • {target.replace('_', ' ').title()}: {min_val} - {max_val}")
    
    print("\n" + "="*60)
    print("✅ Example usage completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
