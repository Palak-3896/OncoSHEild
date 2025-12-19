from gemini_service import GeminiService
import json

class DetectionAgent:
    def __init__(self):
        try:
            self.gemini_service = GeminiService()
        except Exception as e:
            print(f"DetectionAgent: Gemini service initialization failed: {e}")
            self.gemini_service = None
    
    def get_detection_analysis(self, detection_result, image_features=None, user_input=None):
        """Get AI-powered detection analysis"""
        try:
            if self.gemini_service:
                analysis = self.gemini_service.generate_detection_analysis(
                    detection_result, image_features
                )
                if analysis:
                    return analysis
        except Exception as e:
            print(f"DetectionAgent: AI analysis failed: {e}")
        
        # Use fallback analysis
        return self._get_fallback_analysis(detection_result, user_input)
    
    def get_image_analysis(self, detection_result, image_path=None):
        """Get AI analysis specifically for image-based detection"""
        try:
            if self.gemini_service:
                prompt = f"""
                You are a medical AI assistant specializing in breast cancer image analysis and patient guidance.
                
                Detection Result: {detection_result}
                {f"Image Path: {image_path}" if image_path else ""}
                
                Provide comprehensive analysis for image-based detection in JSON format:
                
                {{
                    "image_interpretation": "Explanation of what the image analysis shows",
                    "confidence_assessment": "Assessment of the detection confidence and limitations",
                    "immediate_next_steps": ["List of 4-5 immediate actions to take"],
                    "medical_consultation": {{
                        "urgency_level": "Low/Medium/High urgency description",
                        "specialist_type": "Type of medical specialist to consult",
                        "what_to_expect": "What to expect during medical consultation"
                    }},
                    "additional_testing": ["List of potential additional tests that may be recommended"],
                    "emotional_guidance": ["List of 4-5 emotional support and coping strategies"],
                    "preparation_checklist": ["List of 4-5 items to prepare for medical visit"],
                    "support_resources": ["List of 4-5 support organizations and resources"],
                    "important_reminders": ["List of 3-4 key reminders about the detection process"]
                }}
                """
                
                response = self.gemini_service.model.generate_content(prompt)
                return self.gemini_service._parse_json_response(response.text)
        except Exception as e:
            print(f"DetectionAgent: Image analysis failed: {e}")
        
        return self._get_fallback_image_analysis(detection_result)
    
    def get_numeric_analysis(self, detection_result, numeric_features=None):
        """Get AI analysis for numeric feature-based detection"""
        try:
            if self.gemini_service and numeric_features:
                prompt = f"""
                You are a medical AI assistant specializing in breast cancer detection analysis.
                
                Detection Result: {detection_result}
                Numeric Features: {numeric_features}
                
                Provide detailed analysis for numeric feature-based detection in JSON format:
                
                {{
                    "feature_interpretation": "Explanation of what the numeric features indicate",
                    "risk_factors_identified": ["List of key risk factors from the analysis"],
                    "clinical_significance": "What these measurements mean clinically",
                    "next_steps": {{
                        "immediate_actions": ["List of 3-4 immediate actions"],
                        "medical_follow_up": ["List of 4-5 medical follow-up recommendations"],
                        "monitoring_schedule": "Recommended monitoring frequency"
                    }},
                    "lifestyle_recommendations": {{
                        "preventive_measures": ["List of 4-5 preventive lifestyle measures"],
                        "health_optimization": ["List of 4-5 health optimization tips"],
                        "warning_signs": ["List of warning signs to watch for"]
                    }},
                    "educational_content": {{
                        "understanding_results": "How to understand and interpret the results",
                        "questions_for_doctor": ["List of 5-6 important questions to ask healthcare providers"],
                        "additional_resources": ["List of educational resources"]
                    }},
                    "emotional_support": ["List of 4-5 emotional support strategies"]
                }}
                """
                
                response = self.gemini_service.model.generate_content(prompt)
                return self.gemini_service._parse_json_response(response.text)
        except Exception as e:
            print(f"DetectionAgent: Numeric analysis failed: {e}")
        
        return self._get_fallback_numeric_analysis(detection_result, numeric_features)
    
    def _get_fallback_analysis(self, detection_result, user_input=None):
        """Fallback analysis if AI service fails"""
        if "malignant" in detection_result.lower() or "concerning" in detection_result.lower():
            return {
                "interpretation": "The analysis indicates findings that require immediate medical attention. This screening tool has identified features that may be concerning and warrant further professional evaluation.",
                "immediate_next_steps": [
                    "Contact your healthcare provider immediately to discuss these results",
                    "Schedule an urgent appointment with a breast specialist or oncologist",
                    "Gather all your medical records and family history information",
                    "Prepare a list of current symptoms and concerns"
                ],
                "medical_follow_up": [
                    "Comprehensive clinical breast examination by a specialist",
                    "Additional imaging studies (mammography, ultrasound, or MRI)",
                    "Possible tissue biopsy for definitive diagnosis",
                    "Consultation with multidisciplinary cancer care team",
                    "Discussion of treatment options if diagnosis is confirmed"
                ],
                "emotional_support": [
                    "Remember that early detection significantly improves treatment outcomes",
                    "Seek support from family, friends, or professional counselors",
                    "Consider joining support groups for people facing similar situations",
                    "Practice stress-reduction techniques like meditation or gentle exercise",
                    "Focus on taking one step at a time rather than overwhelming yourself"
                ],
                "preparation_for_medical_visit": [
                    "Compile a comprehensive list of symptoms and their timeline",
                    "Gather complete family medical history, especially cancer history",
                    "List all current medications, supplements, and allergies",
                    "Prepare specific questions about next steps and treatment options",
                    "Consider bringing a trusted friend or family member for support"
                ],
                "support_resources": [
                    "American Cancer Society Helpline: 1-800-227-2345",
                    "National Breast Cancer Foundation (nationalbreastcancer.org)",
                    "Susan G. Komen Foundation support services",
                    "Local hospital cancer support services and social workers",
                    "Online support communities like BreastCancer.org forums"
                ],
                "important_reminders": [
                    "This is a screening tool result, not a final medical diagnosis",
                    "Many concerning findings turn out to be benign upon further testing",
                    "Early detection and prompt treatment lead to the best outcomes",
                    "You have access to excellent medical care and support systems"
                ]
            }
        else:
            return {
                "interpretation": "The analysis suggests findings that appear benign or normal. This is encouraging news, but continued monitoring and regular screening remain important for ongoing breast health.",
                "immediate_next_steps": [
                    "Continue with your regular breast health monitoring routine",
                    "Schedule routine follow-up with your healthcare provider",
                    "Maintain monthly breast self-examinations",
                    "Keep up with recommended screening schedules"
                ],
                "medical_follow_up": [
                    "Follow standard screening guidelines for your age group",
                    "Discuss these results with your healthcare provider at next visit",
                    "Continue regular clinical breast examinations",
                    "Report any new changes or symptoms promptly",
                    "Maintain current preventive care schedule"
                ],
                "emotional_support": [
                    "Feel reassured by these encouraging results",
                    "Continue prioritizing your overall health and wellbeing",
                    "Maintain positive health habits and lifestyle choices",
                    "Stay connected with your healthcare team and support network",
                    "Celebrate taking proactive steps for your health"
                ],
                "preparation_for_medical_visit": [
                    "Discuss these results during your next routine appointment",
                    "Ask about appropriate screening intervals for your situation",
                    "Review your personal and family risk factors",
                    "Update your healthcare provider on any health changes",
                    "Discuss any questions or concerns about breast health"
                ],
                "support_resources": [
                    "American Cancer Society prevention and wellness resources",
                    "National Breast Cancer Foundation educational materials",
                    "Your healthcare provider's patient education resources",
                    "Women's health organizations and screening programs",
                    "Community wellness and prevention programs"
                ],
                "important_reminders": [
                    "Continue regular breast health monitoring and screening",
                    "Encouraging results don't eliminate the need for ongoing care",
                    "Report any new symptoms or changes to your healthcare provider",
                    "Maintain healthy lifestyle choices for continued wellness"
                ]
            }
    
    def _get_fallback_image_analysis(self, detection_result):
        """Fallback image analysis"""
        return {
            "image_interpretation": "The image analysis has been completed using automated detection algorithms. These results should be interpreted by a qualified healthcare professional.",
            "confidence_assessment": "Automated image analysis has limitations and should always be confirmed by medical imaging specialists and clinical examination.",
            "immediate_next_steps": [
                "Share these results with your healthcare provider",
                "Schedule professional medical imaging review",
                "Discuss the need for additional imaging studies",
                "Don't delay seeking professional medical evaluation"
            ],
            "medical_consultation": {
                "urgency_level": "Prompt medical consultation recommended within 1-2 weeks",
                "specialist_type": "Radiologist, breast imaging specialist, or oncologist",
                "what_to_expect": "Professional review of images, possible additional imaging, and clinical correlation"
            },
            "additional_testing": [
                "Professional mammography interpretation",
                "Ultrasound imaging if recommended",
                "Possible MRI for further evaluation",
                "Clinical breast examination"
            ],
            "emotional_guidance": [
                "Remember that many abnormal findings are benign",
                "Seek support from healthcare team and loved ones",
                "Practice stress management techniques",
                "Focus on taking appropriate next steps",
                "Maintain hope and positive outlook"
            ],
            "preparation_checklist": [
                "Gather previous imaging studies for comparison",
                "Prepare list of symptoms and concerns",
                "Research your healthcare provider and facility",
                "Arrange support person to accompany you",
                "Prepare insurance and medical information"
            ],
            "support_resources": [
                "Breast imaging centers and specialists",
                "Patient navigation services",
                "Cancer support organizations",
                "Online educational resources",
                "Peer support groups"
            ],
            "important_reminders": [
                "Automated analysis is a screening tool, not a diagnosis",
                "Professional medical review is essential",
                "Early detection and intervention improve outcomes"
            ]
        }
    
    def _get_fallback_numeric_analysis(self, detection_result, numeric_features):
        """Fallback numeric analysis"""
        feature_summary = "multiple tumor characteristics" if numeric_features else "tumor features"
        
        return {
            "feature_interpretation": f"The analysis of {feature_summary} has been completed. These measurements provide information about tissue characteristics that require professional medical interpretation.",
            "risk_factors_identified": [
                "Tissue characteristics requiring professional evaluation",
                "Measurements outside typical ranges may indicate concern",
                "Combined feature analysis suggests need for medical review",
                "Pattern recognition indicates professional consultation needed"
            ],
            "clinical_significance": "These measurements provide valuable data for healthcare providers to assess tissue characteristics and determine appropriate next steps.",
            "next_steps": {
                "immediate_actions": [
                    "Schedule appointment with healthcare provider",
                    "Bring these results to medical consultation",
                    "Prepare questions about findings and next steps"
                ],
                "medical_follow_up": [
                    "Professional interpretation of measurements",
                    "Correlation with clinical examination",
                    "Possible additional diagnostic testing",
                    "Discussion of treatment options if needed",
                    "Development of monitoring plan"
                ],
                "monitoring_schedule": "Follow healthcare provider recommendations for ongoing monitoring and follow-up care"
            },
            "lifestyle_recommendations": {
                "preventive_measures": [
                    "Maintain regular breast self-examinations",
                    "Follow recommended screening schedules",
                    "Adopt healthy lifestyle choices",
                    "Stay informed about breast health",
                    "Report any changes promptly"
                ],
                "health_optimization": [
                    "Maintain healthy weight and exercise regularly",
                    "Follow nutritious diet rich in antioxidants",
                    "Limit alcohol consumption",
                    "Manage stress effectively",
                    "Get adequate sleep and rest"
                ],
                "warning_signs": [
                    "New lumps or thickening in breast or underarm",
                    "Changes in breast size or shape",
                    "Skin changes like dimpling or puckering",
                    "Nipple discharge or changes"
                ]
            },
            "educational_content": {
                "understanding_results": "These numeric measurements represent various tissue characteristics. Your healthcare provider can explain what these specific values mean for your situation.",
                "questions_for_doctor": [
                    "What do these specific measurements indicate?",
                    "How do these results compare to normal ranges?",
                    "What additional testing might be needed?",
                    "What are the next steps in my care?",
                    "How often should I be monitored?",
                    "Are there lifestyle changes I should make?"
                ],
                "additional_resources": [
                    "Medical literature on breast imaging and diagnosis",
                    "Patient education materials from cancer organizations",
                    "Healthcare provider educational resources"
                ]
            },
            "emotional_support": [
                "Seek support from healthcare team and counselors",
                "Connect with family and friends during this time",
                "Practice relaxation and stress management techniques",
                "Focus on taking appropriate medical steps",
                "Remember that many findings have good outcomes"
            ]
        }