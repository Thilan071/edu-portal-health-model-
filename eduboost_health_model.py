"""
EduBoost Health Recommendation Model

A comprehensive machine learning model for providing personalized health and study 
recommendations to students based on their mood, stress level, procrastination level, 
and sleep hours.

Author: EduBoost Team
Date: August 2025
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from typing import Dict, List, Union, Tuple, Any
import os


class EduBoostHealthModel:
    """
    EduBoost Health Recommendation Model for personalized student wellness predictions.
    
    This model takes student inputs (mood, stress level, procrastination level, sleep hours)
    and predicts optimal health and study recommendations including study hours, exercise minutes,
    sleep hours, water consumption, meditation time, and screen time limits.
    """
    
    def __init__(self, model_path: str = 'eduboost_health_recommendation_model.pkl', verbose: bool = False):
        """
        Initialize the EduBoost Health Model.
        
        Args:
            model_path (str): Path to the trained scikit-learn model pickle file
            verbose (bool): Enable verbose output for detailed logging
        """
        self.model_path = model_path
        self.verbose = verbose
        self.model = None
        self.label_encoder = None
        self.scaler = None
        
        # Valid mood categories
        self.valid_moods = ['Happy', 'Neutral', 'Stressed', 'Sad']
        
        # Prediction bounds (min, max)
        self.bounds = {
            'study_hours': (12, 30),
            'exercise_minutes': (20, 60),
            'sleep_hours': (6.5, 9.5),
            'water_liters': (1.5, 4.0),
            'meditation_minutes': (5, 30),
            'screen_limit': (2, 8)
        }
        
        # Target columns in order
        self.target_columns = [
            'study_hours', 'exercise_minutes', 'sleep_hours', 
            'water_liters', 'meditation_minutes', 'screen_limit'
        ]
        
        # Load the model and preprocessors
        self._load_model()
        self._initialize_preprocessors()
    
    def _load_model(self):
        """Load the trained model from pickle file."""
        try:
            if not os.path.exists(self.model_path):
                if self.verbose:
                    print(f"⚠️  Model file not found: {self.model_path}")
                    print("Will use heuristic predictions instead")
                self.model = None
                return
            
            # Load the model package (includes model, encoders, and scalers)
            model_package = joblib.load(self.model_path)
            
            if isinstance(model_package, dict):
                # New format with preprocessors included
                self.model = model_package['model']
                self.label_encoder = model_package['label_encoder']
                self.scaler = model_package['scaler']
                
                if self.verbose:
                    print(f"✓ Model package loaded successfully from {self.model_path}")
                    print(f"Model type: {type(self.model).__name__}")
                    if 'model_info' in model_package:
                        print(f"Training date: {model_package['model_info'].get('training_date', 'Unknown')}")
            else:
                # Old format (just the model)
                self.model = model_package
                if self.verbose:
                    print(f"✓ Legacy model loaded from {self.model_path}")
                    print(f"Model type: {type(self.model).__name__}")
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to load model: {str(e)}")
                print("Will use heuristic predictions instead")
            self.model = None
    
    def _initialize_preprocessors(self):
        """Initialize the label encoder and standard scaler."""
        # Only initialize if not already loaded from model package
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.valid_moods)
        
        if self.scaler is None:
            self.scaler = StandardScaler()
        
        if self.verbose:
            print("✓ Preprocessors initialized")
            if hasattr(self.label_encoder, 'classes_'):
                print(f"Valid moods: {list(self.label_encoder.classes_)}")
            else:
                print(f"Valid moods: {self.valid_moods}")
    
    def _validate_inputs(self, mood: str, stress_level: int, procrastination_level: int, sleep_hours: float):
        """
        Validate all input parameters.
        
        Args:
            mood (str): Student's mood
            stress_level (int): Stress level (1-5)
            procrastination_level (int): Procrastination level (1-5)
            sleep_hours (float): Hours of sleep
            
        Raises:
            ValueError: If any input is invalid
        """
        # Validate mood
        if not isinstance(mood, str):
            raise ValueError("Mood must be a string")
        
        if mood not in self.valid_moods:
            raise ValueError(f"Invalid mood '{mood}'. Must be one of: {self.valid_moods}")
        
        # Validate stress level
        if not isinstance(stress_level, (int, float)):
            raise ValueError("Stress level must be a number")
        
        if not (1 <= stress_level <= 5):
            raise ValueError("Stress level must be between 1 and 5")
        
        # Validate procrastination level
        if not isinstance(procrastination_level, (int, float)):
            raise ValueError("Procrastination level must be a number")
        
        if not (1 <= procrastination_level <= 5):
            raise ValueError("Procrastination level must be between 1 and 5")
        
        # Validate sleep hours
        if not isinstance(sleep_hours, (int, float)):
            raise ValueError("Sleep hours must be a number")
        
        if not (0 <= sleep_hours <= 24):
            raise ValueError("Sleep hours must be between 0 and 24")
        
        if self.verbose:
            print("✓ All inputs validated successfully")
    
    def _encode_and_scale_features(self, mood: str, stress_level: float, 
                                 procrastination_level: float, sleep_hours: float) -> np.ndarray:
        """
        Encode mood and scale all features.
        
        Args:
            mood (str): Student's mood
            stress_level (float): Stress level
            procrastination_level (float): Procrastination level
            sleep_hours (float): Hours of sleep
            
        Returns:
            np.ndarray: Encoded and scaled features
        """
        # Encode mood
        mood_encoded = self.label_encoder.transform([mood])[0]
        
        # Create feature array
        features = np.array([[mood_encoded, stress_level, procrastination_level, sleep_hours]])
        
        # Scale features (use pre-fitted scaler if available)
        if hasattr(self.scaler, 'scale_') and self.scaler.scale_ is not None:
            # Scaler is already fitted (loaded from model package)
            features_scaled = self.scaler.transform(features)
        else:
            # Fit scaler with dummy data (fallback for legacy models)
            dummy_data = np.array([
                [0, 1, 1, 0],    # min values
                [3, 5, 5, 24]    # max values (mood encoded 0-3, others 1-5, sleep 0-24)
            ])
            self.scaler.fit(dummy_data)
            features_scaled = self.scaler.transform(features)
        
        if self.verbose:
            print(f"✓ Features encoded and scaled: {features_scaled[0]}")
        
        return features_scaled
    
    def _apply_bounds(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply realistic bounds to predictions.
        
        Args:
            predictions (np.ndarray): Raw model predictions
            
        Returns:
            np.ndarray: Bounded predictions
        """
        bounded_predictions = predictions.copy()
        
        for i, target in enumerate(self.target_columns):
            min_val, max_val = self.bounds[target]
            bounded_predictions[i] = np.clip(bounded_predictions[i], min_val, max_val)
        
        if self.verbose:
            print("✓ Bounds applied to predictions")
        
        return bounded_predictions
    
    def _generate_personalized_plans(self, mood: str, stress_level: int, procrastination_level: int, 
                                    sleep_hours: float, predictions: Dict[str, float]) -> Dict[str, str]:
        """
        Generate comprehensive personalized text plans based on inputs and predictions.
        
        Args:
            mood (str): Student's mood
            stress_level (int): Student's stress level (1-5)
            procrastination_level (int): Student's procrastination level (1-5)
            sleep_hours (float): Student's current sleep hours
            predictions (dict): Model predictions
            
        Returns:
            dict: Comprehensive plans for study, physical, and emotional wellness
        """
        study_hours = predictions['study_hours']
        exercise_minutes = predictions['exercise_minutes']
        sleep_target = predictions['sleep_hours']
        water_target = predictions['water_liters']
        meditation_minutes = predictions['meditation_minutes']
        screen_limit = predictions['screen_limit']
        
        # Weekly Study Plan based on stress, procrastination, and predicted hours
        if stress_level >= 4 and procrastination_level >= 4:
            # High stress + High procrastination
            study_plan = (f"**Weekly Study Plan - Intensive Focus Strategy**\n"
                         f"Recommended Study Hours: {study_hours:.0f} hours/week\n"
                         f"• Use Pomodoro Technique: 25-min study, 5-min break cycles\n"
                         f"• Schedule 3-4 study sessions daily (morning preferred)\n"
                         f"• Break large tasks into 15-minute mini-sessions\n"
                         f"• Use accountability partner or study group\n"
                         f"• Reward completion of each session\n"
                         f"• Focus on one subject per session to avoid overwhelm")
        elif stress_level >= 3 or procrastination_level >= 3:
            # Moderate stress or procrastination
            study_plan = (f"**Weekly Study Plan - Balanced Approach**\n"
                         f"Recommended Study Hours: {study_hours:.0f} hours/week\n"
                         f"• Organize study time into 45-60 minute blocks\n"
                         f"• Take 15-minute breaks between sessions\n"
                         f"• Mix challenging and easier subjects daily\n"
                         f"• Schedule review sessions twice weekly\n"
                         f"• Use active learning techniques (flashcards, practice tests)\n"
                         f"• Plan study schedule at beginning of each week")
        else:
            # Low stress + Low procrastination
            study_plan = (f"**Weekly Study Plan - Deep Learning Focus**\n"
                         f"Recommended Study Hours: {study_hours:.0f} hours/week\n"
                         f"• Enjoy 90-120 minute deep learning sessions\n"
                         f"• Take 20-30 minute breaks for reflection\n"
                         f"• Explore topics that genuinely interest you\n"
                         f"• Focus on understanding concepts deeply\n"
                         f"• Engage in research and creative projects\n"
                         f"• Allow flexible scheduling based on energy levels")
        
        # Physical Plan based on stress level and mood
        if stress_level >= 4:
            # High stress - need intensive physical activity
            if mood == 'Stressed':
                physical_plan = (f"**Physical Plan - Stress Relief Focus**\n"
                               f"• High-intensity workout {exercise_minutes:.0f} mins daily (running, HIIT, boxing)\n"
                               f"• Minimum {sleep_target:.1f} hours sleep nightly\n"
                               f"• Drink {water_target:.1f}L water daily (track with app)\n"
                               f"• 10 minutes stretching after intense workouts\n"
                               f"• Cold shower or breathing exercises post-workout\n"
                               f"• Schedule workouts when stress peaks (often evenings)")
            else:
                physical_plan = (f"**Physical Plan - Energy Release**\n"
                               f"• Vigorous cardio {exercise_minutes:.0f} mins daily (cycling, swimming, dancing)\n"
                               f"• Target {sleep_target:.1f} hours quality sleep\n"
                               f"• Hydrate with {water_target:.1f}L water throughout day\n"
                               f"• Include strength training 3x per week\n"
                               f"• Practice yoga or tai chi for balance\n"
                               f"• Weekend outdoor activities (hiking, sports)")
        elif stress_level >= 2:
            # Moderate stress
            physical_plan = (f"**Physical Plan - Balanced Fitness**\n"
                           f"• Moderate exercise {exercise_minutes:.0f} mins daily (brisk walk, gym, sports)\n"
                           f"• Consistent {sleep_target:.1f} hours sleep schedule\n"
                           f"• Maintain {water_target:.1f}L daily water intake\n"
                           f"• Mix cardio (20 min) and strength training (20 min)\n"
                           f"• Add flexibility work (yoga, stretching) 3x weekly\n"
                           f"• Choose activities you genuinely enjoy")
        else:
            # Low stress
            physical_plan = (f"**Physical Plan - Gentle Wellness**\n"
                           f"• Light movement {exercise_minutes:.0f} mins daily (walking, gentle yoga, swimming)\n"
                           f"• Relaxed {sleep_target:.1f} hours sleep routine\n"
                           f"• Gentle hydration with {water_target:.1f}L water daily\n"
                           f"• Focus on activities that bring joy\n"
                           f"• Nature walks or outdoor meditation\n"
                           f"• Listen to music or podcasts during exercise")
        
        # Emotional Plan based on mood, stress, and sleep quality
        sleep_quality = "good" if sleep_hours >= 7 else "needs improvement"
        
        if mood == 'Sad' or stress_level >= 4:
            # Need emotional support
            emotional_plan = (f"**Emotional Plan - Mental Health Priority**\n"
                            f"• Practice mindfulness meditation {meditation_minutes:.0f} mins daily\n"
                            f"• Limit screen time {screen_limit:.1f} hours before sleep\n"
                            f"• Schedule weekly mental health check-ins with counselor\n"
                            f"• Journal thoughts and feelings daily (10-15 mins)\n"
                            f"• Connect with supportive friends/family 3x weekly\n"
                            f"• Practice gratitude exercises before bed\n"
                            f"• Consider professional support if feelings persist")
        elif mood == 'Stressed':
            # Stress management focus
            emotional_plan = (f"**Emotional Plan - Stress Management**\n"
                            f"• Guided meditation or breathing exercises {meditation_minutes:.0f} mins daily\n"
                            f"• Screen curfew {screen_limit:.1f} hours before bedtime\n"
                            f"• Weekly mentor or trusted friend check-ins\n"
                            f"• Stress journaling twice weekly\n"
                            f"• Progressive muscle relaxation before sleep\n"
                            f"• Identify and avoid stress triggers when possible\n"
                            f"• Practice saying 'no' to overwhelming commitments")
        elif mood == 'Happy':
            # Maintain positive state
            emotional_plan = (f"**Emotional Plan - Positive Momentum**\n"
                            f"• Mindfulness practice {meditation_minutes:.0f} mins daily for awareness\n"
                            f"• Healthy screen boundary {screen_limit:.1f} hours before sleep\n"
                            f"• Share positive experiences with friends weekly\n"
                            f"• Celebrate small wins and achievements\n"
                            f"• Engage in creative activities or hobbies\n"
                            f"• Practice acts of kindness toward others\n"
                            f"• Maintain gratitude practice")
        else:  # Neutral
            # Balanced emotional wellness
            emotional_plan = (f"**Emotional Plan - Balanced Wellness**\n"
                            f"• Regular meditation or mindfulness {meditation_minutes:.0f} mins daily\n"
                            f"• Digital detox {screen_limit:.1f} hours before bed\n"
                            f"• Weekly reflection sessions with mentor or journal\n"
                            f"• Social connections 2-3 times per week\n"
                            f"• Explore new interests or learning opportunities\n"
                            f"• Practice emotional awareness throughout day\n"
                            f"• Maintain work-life balance boundaries")
        
        return {
            'study_plan': study_plan,
            'physical_plan': physical_plan,
            'emotional_plan': emotional_plan
        }
    
    def _calculate_confidence(self, features: np.ndarray) -> float:
        """
        Calculate model confidence based on feature values.
        
        Args:
            features (np.ndarray): Scaled input features
            
        Returns:
            float: Confidence score (0-1)
        """
        # Simple confidence calculation based on feature normality
        # In practice, this could use prediction intervals or ensemble variance
        
        # Check if features are within reasonable ranges after scaling
        feature_confidence = 1.0 - np.mean(np.abs(features[0]))
        
        # Ensure confidence is between 0.6 and 0.95
        confidence = np.clip(feature_confidence, 0.6, 0.95)
        
        return confidence
    
    def predict(self, mood: str, stress_level: int, procrastination_level: int, 
                sleep_hours: float) -> Dict[str, Any]:
        """
        Make a health and study recommendation prediction for a single student.
        
        Args:
            mood (str): Student's mood ('Happy', 'Neutral', 'Stressed', 'Sad')
            stress_level (int): Stress level (1-5)
            procrastination_level (int): Procrastination level (1-5)
            sleep_hours (float): Hours of sleep
            
        Returns:
            dict: Comprehensive prediction results including analysis, recommendations, 
                  metrics, and confidence
        """
        if self.verbose:
            print(f"\n🎯 Making prediction for student...")
            print(f"Inputs - Mood: {mood}, Stress: {stress_level}, Procrastination: {procrastination_level}, Sleep: {sleep_hours}h")
        
        try:
            # Validate inputs
            self._validate_inputs(mood, stress_level, procrastination_level, sleep_hours)
            
            # Encode and scale features
            features_scaled = self._encode_and_scale_features(
                mood, stress_level, procrastination_level, sleep_hours
            )
            
            # Make prediction (handle case where model might not exist)
            if self.model is None:
                # Use simple heuristic predictions if model is not available
                warnings.warn("Model not available, using heuristic predictions")
                raw_predictions = self._heuristic_predictions(stress_level, procrastination_level, sleep_hours)
            else:
                raw_predictions = self.model.predict(features_scaled)[0]
            
            # Apply bounds
            bounded_predictions = self._apply_bounds(raw_predictions)
            
            # Create predictions dictionary
            predictions_dict = {
                target: float(bounded_predictions[i]) 
                for i, target in enumerate(self.target_columns)
            }
            
            # Generate personalized plans
            personalized_plans = self._generate_personalized_plans(
                mood, stress_level, procrastination_level, sleep_hours, predictions_dict
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(features_scaled)
            
            # Prepare comprehensive result
            result = {
                'input_analysis': {
                    'mood': mood,
                    'stress_level': stress_level,
                    'procrastination_level': procrastination_level,
                    'sleep_hours': sleep_hours,
                    'stress_category': self._categorize_stress(stress_level),
                    'sleep_quality': self._assess_sleep_quality(sleep_hours)
                },
                'recommendations': personalized_plans,
                'detailed_metrics': {
                    'study_hours': round(predictions_dict['study_hours'], 1),
                    'exercise_minutes': round(predictions_dict['exercise_minutes'], 0),
                    'sleep_hours': round(predictions_dict['sleep_hours'], 1),
                    'water_liters': round(predictions_dict['water_liters'], 1),
                    'meditation_minutes': round(predictions_dict['meditation_minutes'], 0),
                    'screen_limit': round(predictions_dict['screen_limit'], 1)
                },
                'model_confidence': round(confidence, 3)
            }
            
            if self.verbose:
                print("✓ Prediction completed successfully")
                print(f"Confidence: {confidence:.3f}")
            
            return result
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    def _heuristic_predictions(self, stress_level: int, procrastination_level: int, sleep_hours: float) -> np.ndarray:
        """
        Generate heuristic predictions when model is not available.
        
        Args:
            stress_level (int): Stress level (1-5)
            procrastination_level (int): Procrastination level (1-5)
            sleep_hours (float): Hours of sleep
            
        Returns:
            np.ndarray: Heuristic predictions for all targets
        """
        # Simple heuristic based on stress and procrastination
        base_study = 20 - (stress_level * 1.5) - (procrastination_level * 1.0)
        base_exercise = 30 + (stress_level * 5)
        base_sleep = 8.0 - (stress_level * 0.3)
        base_water = 2.5 + (stress_level * 0.2)
        base_meditation = 10 + (stress_level * 3)
        base_screen = 6 - (stress_level * 0.5)
        
        return np.array([base_study, base_exercise, base_sleep, base_water, base_meditation, base_screen])
    
    def _categorize_stress(self, stress_level: int) -> str:
        """Categorize stress level into descriptive text."""
        if stress_level <= 2:
            return "Low stress - well managed"
        elif stress_level <= 3:
            return "Moderate stress - manageable"
        elif stress_level <= 4:
            return "High stress - needs attention"
        else:
            return "Very high stress - requires immediate support"
    
    def _assess_sleep_quality(self, sleep_hours: float) -> str:
        """Assess sleep quality based on hours."""
        if sleep_hours < 6:
            return "Insufficient sleep - health risk"
        elif sleep_hours < 7:
            return "Below optimal - may affect performance"
        elif sleep_hours <= 9:
            return "Good sleep duration"
        else:
            return "Extended sleep - may indicate fatigue"
    
    def batch_predict(self, students_data: List[Dict[str, Union[str, int, float]]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple students.
        
        Args:
            students_data (list): List of dictionaries containing student data
                Each dict should have keys: 'mood', 'stress_level', 'procrastination_level', 'sleep_hours'
        
        Returns:
            list: List of prediction results for each student
        """
        if self.verbose:
            print(f"\n📊 Processing batch prediction for {len(students_data)} students...")
        
        results = []
        
        for i, student_data in enumerate(students_data):
            try:
                if self.verbose:
                    print(f"Processing student {i+1}/{len(students_data)}")
                
                result = self.predict(
                    mood=student_data['mood'],
                    stress_level=student_data['stress_level'],
                    procrastination_level=student_data['procrastination_level'],
                    sleep_hours=student_data['sleep_hours']
                )
                
                # Add student identifier
                result['student_id'] = i + 1
                results.append(result)
                
            except Exception as e:
                error_result = {
                    'student_id': i + 1,
                    'error': str(e),
                    'input_data': student_data
                }
                results.append(error_result)
                
                if self.verbose:
                    print(f"❌ Error processing student {i+1}: {str(e)}")
        
        if self.verbose:
            successful_predictions = len([r for r in results if 'error' not in r])
            print(f"✓ Batch prediction completed: {successful_predictions}/{len(students_data)} successful")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model and preprocessors.
        
        Returns:
            dict: Model information including type, bounds, and valid inputs
        """
        return {
            'model_path': self.model_path,
            'model_type': type(self.model).__name__ if self.model else 'Not loaded',
            'valid_moods': self.valid_moods,
            'target_columns': self.target_columns,
            'prediction_bounds': self.bounds,
            'has_label_encoder': self.label_encoder is not None,
            'has_scaler': self.scaler is not None,
            'verbose_mode': self.verbose
        }


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the EduBoost Health Model
    
    # Note: This will use heuristic predictions since the actual model file doesn't exist
    print("🎓 EduBoost Health Recommendation Model Demo")
    print("=" * 50)
    
    try:
        # Initialize model (will use heuristics if model file doesn't exist)
        model = EduBoostHealthModel(verbose=True)
        
        # Single prediction example
        print("\n📝 Single Student Prediction:")
        result = model.predict(
            mood='Stressed',
            stress_level=4,
            procrastination_level=3,
            sleep_hours=6.5
        )
        
        print("\n📊 Results:")
        print(f"Confidence: {result['model_confidence']}")
        print(f"Study Hours: {result['detailed_metrics']['study_hours']}")
        print(f"Exercise: {result['detailed_metrics']['exercise_minutes']} minutes")
        print(f"Sleep: {result['detailed_metrics']['sleep_hours']} hours")
        print(f"Water: {result['detailed_metrics']['water_liters']} liters")
        print(f"Meditation: {result['detailed_metrics']['meditation_minutes']} minutes")
        print(f"Screen limit: {result['detailed_metrics']['screen_limit']} hours")
        
        print(f"\n📚 Study Plan: {result['recommendations']['study_plan'][:100]}...")
        print(f"💪 Physical Plan: {result['recommendations']['physical_plan'][:100]}...")
        print(f"🧘 Emotional Plan: {result['recommendations']['emotional_plan'][:100]}...")
        
        # Batch prediction example
        print("\n\n👥 Batch Prediction Example:")
        students = [
            {'mood': 'Happy', 'stress_level': 2, 'procrastination_level': 2, 'sleep_hours': 8.0},
            {'mood': 'Stressed', 'stress_level': 5, 'procrastination_level': 4, 'sleep_hours': 5.5},
            {'mood': 'Neutral', 'stress_level': 3, 'procrastination_level': 3, 'sleep_hours': 7.5}
        ]
        
        batch_results = model.batch_predict(students)
        
        for i, result in enumerate(batch_results):
            if 'error' not in result:
                print(f"Student {i+1}: {result['detailed_metrics']['study_hours']}h study, "
                      f"{result['detailed_metrics']['exercise_minutes']}min exercise, "
                      f"confidence: {result['model_confidence']}")
        
        # Model info
        print(f"\n🔧 Model Info:")
        info = model.get_model_info()
        print(f"Model type: {info['model_type']}")
        print(f"Valid moods: {info['valid_moods']}")
        print(f"Targets: {info['target_columns']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
