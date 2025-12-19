import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GeminiService:
    def __init__(self):
        # Configure Gemini API
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('models/gemma-3-12b-it')
    
    def generate_risk_recommendations(self, risk_level, risk_score, input_data):
        """Generate comprehensive AI recommendations for breast cancer risk assessment"""
        
        prompt = f"""
        You are a healthcare AI assistant specializing in breast cancer risk assessment and prevention. 
        
        Patient Profile:
        - Risk Level: {risk_level}
        - Risk Score: {risk_score}%
        - Age: {input_data.get('Age', 'N/A')} years
        - Family History: {input_data.get('Family_History', 'N/A')}
        - Genetic Mutation: {input_data.get('Genetic_Mutation', 'N/A')}
        - BMI: {input_data.get('BMI', 'N/A')}
        - Previous Biopsy: {input_data.get('Previous_Biopsy', 'N/A')}
        
        Based on this risk assessment, provide comprehensive recommendations in the following JSON format. For low risk cases, ensure to start your diet recommendations with a reassuring but informative message about the importance of prevention even with low risk.
        
        {{
            "refined_analysis": "A detailed analysis of the risk factors and what they mean",
            "immediate_actions": ["List of 3-4 immediate actions the person should take"],
            "diet_recommendations": {{
                "introduction": "For low risk patients, provide a reassuring message about prevention while emphasizing the importance of maintaining healthy dietary habits. For moderate/high risk patients, emphasize how these dietary changes can help manage risk factors.",
                "foods_to_include": ["List of 5-6 specific foods/food groups to include"],
                "foods_to_avoid": ["List of 4-5 foods/substances to limit or avoid"],
                "meal_planning_tips": ["List of 4-5 practical meal planning tips"]
            }},
            "lifestyle_changes": {{
                "exercise_recommendations": ["List of 4-5 specific exercise recommendations"],
                "stress_management": ["List of 3-4 stress management techniques"],
                "sleep_hygiene": ["List of 3-4 sleep improvement tips"]
            }},
            "medical_care": {{
                "screening_schedule": "Recommended screening frequency and timing",
                "specialist_consultations": ["List of 2-3 types of specialists to consider"],
                "questions_for_doctor": ["List of 4-5 important questions to ask healthcare providers"]
            }},
            "monitoring_and_tracking": ["List of 4-5 things to monitor or track"],
            "support_resources": ["List of 4-5 support resources or organizations"]
        }}
        
        Ensure all recommendations are:
        - Evidence-based and medically appropriate
        - Personalized to the specific risk level and factors
        - Actionable and practical
        - Encouraging but realistic
        - Include disclaimers about consulting healthcare professionals
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return None
    
    def generate_detection_analysis(self, detection_result, image_features=None):
        """Generate AI analysis for breast cancer detection results"""
        
        is_non_cancerous = "non" in detection_result.lower() or "benign" in detection_result.lower()
        
        prompt = f"""
        You are a healthcare AI assistant specializing in breast cancer detection and patient support.
        
        Detection Result: {detection_result}
        {f"Image Features: {image_features}" if image_features else ""}
        
        Provide comprehensive analysis and guidance in the following JSON format. For non-cancerous/benign results, ensure to start with a reassuring message while still emphasizing the importance of regular monitoring and preventive care.
        
        {{
            "initial_message": "For non-cancerous results: Start with a positive, reassuring message about the benign finding while emphasizing the importance of continued vigilance. For concerning results: Focus on the importance of prompt follow-up while maintaining a supportive tone.",
            "interpretation": "Clear explanation of what the detection result means",
            "immediate_next_steps": ["List of 3-4 immediate actions to take"],
            "medical_follow_up": ["List of 4-5 specific medical follow-up recommendations"],
            "support_resources": ["List of 4-5 support organizations and resources"],
            "important_reminders": ["List of 3-4 important reminders and reassurances"],
            "preventive_recommendations": {{
                "introduction": "For non-cancerous results: Emphasize maintaining good breast health practices. For concerning results: Focus on proactive monitoring and risk reduction.",
                "lifestyle_tips": ["List of 3-4 lifestyle recommendations"],
                "monitoring_guidelines": ["List of 3-4 specific monitoring recommendations"]
            }}
        }}
        
        Ensure the response is:
        - Compassionate and supportive
        - Medically accurate
        - Emphasizes the importance of professional medical evaluation
        - Provides practical next steps and medical guidance
        - Includes relevant support resources
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            print(f"Error generating detection analysis: {e}")
            return None
    
    def _parse_json_response(self, response_text):
        """Parse JSON response from Gemini, handling potential formatting issues"""
        import json
        import re
        
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # If no JSON found, return None
                return None
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return None
        except Exception as e:
            print(f"General parsing error: {e}")
            return None