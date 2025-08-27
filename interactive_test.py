"""
Interactive EduBoost Health Model Test Script

This script allows you to test the EduBoost Health Model with your own inputs
interactively through the terminal.
"""

import sys
import os
from eduboost_health_model import EduBoostHealthModel


def get_user_input():
    """Get user inputs interactively."""
    print("🎓 EduBoost Health Model - Interactive Test")
    print("=" * 50)
    print("Please provide your current status:")
    print()
    
    # Get mood
    print("💭 Available moods: Happy, Neutral, Stressed, Sad")
    while True:
        mood = input("Enter your current mood: ").strip()
        if mood in ['Happy', 'Neutral', 'Stressed', 'Sad']:
            break
        print("❌ Invalid mood. Please choose from: Happy, Neutral, Stressed, Sad")
    
    # Get stress level
    while True:
        try:
            stress_level = int(input("Enter your stress level (1-5, where 1=very low, 5=very high): "))
            if 1 <= stress_level <= 5:
                break
            print("❌ Stress level must be between 1 and 5")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Get procrastination level
    while True:
        try:
            procrastination_level = int(input("Enter your procrastination level (1-5, where 1=very low, 5=very high): "))
            if 1 <= procrastination_level <= 5:
                break
            print("❌ Procrastination level must be between 1 and 5")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Get sleep hours
    while True:
        try:
            sleep_hours = float(input("Enter how many hours you slept last night (0-24): "))
            if 0 <= sleep_hours <= 24:
                break
            print("❌ Sleep hours must be between 0 and 24")
        except ValueError:
            print("❌ Please enter a valid number")
    
    return mood, stress_level, procrastination_level, sleep_hours


def display_results(result):
    """Display the prediction results in a user-friendly format."""
    print("\n" + "=" * 60)
    print("🎯 YOUR PERSONALIZED HEALTH RECOMMENDATIONS")
    print("=" * 60)
    
    # Input Analysis
    print("\n📊 INPUT ANALYSIS:")
    print("-" * 30)
    analysis = result['input_analysis']
    print(f"Mood: {analysis['mood']}")
    print(f"Stress Level: {analysis['stress_level']} ({analysis['stress_category']})")
    print(f"Procrastination Level: {analysis['procrastination_level']}")
    print(f"Sleep Hours: {analysis['sleep_hours']} ({analysis['sleep_quality']})")
    print(f"Model Confidence: {result['model_confidence'] * 100:.1f}%")
    
    # Detailed Metrics
    print("\n📈 RECOMMENDED METRICS:")
    print("-" * 30)
    metrics = result['detailed_metrics']
    print(f"📚 Study Hours per Week: {metrics['study_hours']} hours")
    print(f"🏃 Exercise per Day: {metrics['exercise_minutes']} minutes")
    print(f"😴 Sleep per Night: {metrics['sleep_hours']} hours")
    print(f"💧 Water per Day: {metrics['water_liters']} liters")
    print(f"🧘 Meditation per Day: {metrics['meditation_minutes']} minutes")
    print(f"📱 Screen Time Limit: {metrics['screen_limit']} hours before bed")
    
    # Study Plan
    print("\n📚 YOUR STUDY PLAN:")
    print("-" * 30)
    print(result['recommendations']['study_plan'])
    
    # Physical Plan
    print("\n💪 YOUR PHYSICAL PLAN:")
    print("-" * 30)
    print(result['recommendations']['physical_plan'])
    
    # Emotional Plan
    print("\n🧘 YOUR EMOTIONAL WELLNESS PLAN:")
    print("-" * 30)
    print(result['recommendations']['emotional_plan'])


def main():
    """Main interactive function."""
    try:
        # Initialize the model
        print("🔄 Loading EduBoost Health Model...")
        model = EduBoostHealthModel(verbose=False)  # Set to True if you want detailed logs
        print("✅ Model loaded successfully!")
        print()
        
        while True:
            try:
                # Get user inputs
                mood, stress_level, procrastination_level, sleep_hours = get_user_input()
                
                print("\n🔄 Processing your inputs...")
                
                # Make prediction
                result = model.predict(
                    mood=mood,
                    stress_level=stress_level,
                    procrastination_level=procrastination_level,
                    sleep_hours=sleep_hours
                )
                
                # Display results
                display_results(result)
                
                # Ask if user wants to test again
                print("\n" + "=" * 60)
                while True:
                    continue_test = input("Would you like to test with different inputs? (y/n): ").strip().lower()
                    if continue_test in ['y', 'yes']:
                        print("\n" + "="*60)
                        break
                    elif continue_test in ['n', 'no']:
                        print("\n👋 Thank you for using EduBoost Health Model!")
                        print("Stay healthy and keep learning! 🎓✨")
                        return
                    else:
                        print("❌ Please enter 'y' for yes or 'n' for no")
                
            except KeyboardInterrupt:
                print("\n\n👋 Test interrupted by user. Goodbye!")
                return
            except Exception as e:
                print(f"\n❌ Error during prediction: {str(e)}")
                print("Please try again with valid inputs.")
                continue
                
    except Exception as e:
        print(f"❌ Failed to initialize model: {str(e)}")
        print("Please make sure the model file exists or check your setup.")
        return


if __name__ == "__main__":
    main()
