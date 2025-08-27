"""
Updated EduBoost Health Model - Comprehensive Plan Generation

This script demonstrates the enhanced model that generates detailed text-based
weekly study plans, physical plans, and emotional plans based on user inputs.
"""

from eduboost_health_model import EduBoostHealthModel


def main():
    """Demonstrate the updated EduBoost Health Model with comprehensive plan generation."""
    
    print("🎓 EduBoost Health Model - Enhanced Plan Generation")
    print("=" * 70)
    
    # Initialize the model
    try:
        model = EduBoostHealthModel(verbose=False)
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"⚠️  Model initialization: {e}")
        model = EduBoostHealthModel(verbose=False)
    
    # Test cases covering different combinations
    test_cases = [
        {
            'name': 'High Stress Student',
            'mood': 'Stressed',
            'stress_level': 5,
            'procrastination_level': 4,
            'sleep_hours': 5.5
        },
        {
            'name': 'Happy Balanced Student',
            'mood': 'Happy',
            'stress_level': 2,
            'procrastination_level': 2,
            'sleep_hours': 8.0
        },
        {
            'name': 'Sad Struggling Student',
            'mood': 'Sad',
            'stress_level': 4,
            'procrastination_level': 5,
            'sleep_hours': 6.0
        },
        {
            'name': 'Neutral Moderate Student',
            'mood': 'Neutral',
            'stress_level': 3,
            'procrastination_level': 3,
            'sleep_hours': 7.5
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n" + "="*70)
        print(f"📋 EXAMPLE {i}: {case['name']}")
        print("="*70)
        print(f"Input: {case['mood']} mood, Stress: {case['stress_level']}/5, "
              f"Procrastination: {case['procrastination_level']}/5, Sleep: {case['sleep_hours']}h")
        
        try:
            result = model.predict(
                mood=case['mood'],
                stress_level=case['stress_level'],
                procrastination_level=case['procrastination_level'],
                sleep_hours=case['sleep_hours']
            )
            
            # Display the comprehensive plans
            print(f"\n📚 {result['recommendations']['study_plan']}")
            print(f"\n💪 {result['recommendations']['physical_plan']}")
            print(f"\n🧘 {result['recommendations']['emotional_plan']}")
            
            # Show quick metrics
            metrics = result['detailed_metrics']
            print(f"\n📊 Quick Metrics Summary:")
            print(f"   Study: {metrics['study_hours']}h/week | Exercise: {metrics['exercise_minutes']}min/day")
            print(f"   Sleep: {metrics['sleep_hours']}h/night | Water: {metrics['water_liters']}L/day")
            print(f"   Meditation: {metrics['meditation_minutes']}min/day | Screen limit: {metrics['screen_limit']}h before bed")
            print(f"   Model Confidence: {result['model_confidence']:.1%}")
            
        except Exception as e:
            print(f"❌ Error generating plan: {e}")
    
    # Interactive example
    print(f"\n" + "="*70)
    print("🎯 INTERACTIVE EXAMPLE")
    print("="*70)
    print("Let's try with a custom input...")
    
    # Example custom input
    custom_case = {
        'mood': 'Happy',
        'stress_level': 1,
        'procrastination_level': 1,
        'sleep_hours': 8.5
    }
    
    print(f"Custom Input: {custom_case['mood']} mood, Very low stress and procrastination, Great sleep")
    
    try:
        result = model.predict(**custom_case)
        
        print(f"\n🎉 RESULT FOR WELL-BALANCED STUDENT:")
        print(f"\n{result['recommendations']['study_plan']}")
        print(f"\n{result['recommendations']['physical_plan']}")
        print(f"\n{result['recommendations']['emotional_plan']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Summary
    print(f"\n" + "="*70)
    print("✅ ENHANCED MODEL FEATURES")
    print("="*70)
    print("🔸 Comprehensive weekly study plans based on stress & procrastination")
    print("🔸 Detailed physical plans considering mood and stress levels")
    print("🔸 Personalized emotional wellness plans for mental health")
    print("🔸 Specific recommendations with actionable steps")
    print("🔸 Text-based outputs ready for student implementation")
    print("🔸 Considers all 4 input factors: mood, stress, procrastination, sleep")
    
    print(f"\n🎓 The enhanced EduBoost Health Model provides comprehensive,")
    print(f"   actionable wellness plans tailored to each student's unique situation!")


if __name__ == "__main__":
    main()
