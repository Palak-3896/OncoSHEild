from gemini_service import GeminiService
import json

class AIAgent:
    def __init__(self):
        self.gemini_service = GeminiService()
    
    def get_risk_recommendations(self, risk_level, risk_score, input_data):
        """Get AI-powered risk recommendations"""
        try:
            recommendations = self.gemini_service.generate_risk_recommendations(
                risk_level, risk_score, input_data
            )
            return recommendations
        except Exception as e:
            print(f"Error getting risk recommendations: {e}")
            return self._get_fallback_recommendations(risk_level)
    
    def get_detection_analysis(self, detection_result, image_features=None):
        """Get AI-powered detection analysis"""
        try:
            analysis = self.gemini_service.generate_detection_analysis(
                detection_result, image_features
            )
            return analysis
        except Exception as e:
            print(f"Error getting detection analysis: {e}")
            return self._get_fallback_analysis(detection_result)
    
    def _get_fallback_recommendations(self, risk_level):
        """Fallback recommendations if AI service fails"""
        if "High Risk" in risk_level:
            return {
                "refined_analysis": "Based on your risk factors, you may have an elevated risk of breast cancer. This assessment considers multiple factors including family history, genetic factors, and lifestyle elements.",
                "immediate_actions": [
                    "Schedule an appointment with your healthcare provider to discuss these results",
                    "Consider genetic counseling if you have strong family history",
                    "Discuss appropriate screening schedule with your doctor",
                    "Start implementing lifestyle modifications for risk reduction"
                ],
                "diet_recommendations": {
                    "foods_to_include": [
                        "Leafy green vegetables (spinach, kale, arugula)",
                        "Antioxidant-rich berries (blueberries, strawberries, raspberries)",
                        "Omega-3 rich fish (salmon, mackerel, sardines)",
                        "Nuts and seeds (walnuts, flaxseeds, chia seeds)",
                        "Colorful vegetables (carrots, bell peppers, broccoli)",
                        "Whole grains and legumes"
                    ],
                    "foods_to_avoid": [
                        "Processed and red meats",
                        "Refined sugars and processed foods",
                        "Excessive alcohol consumption",
                        "High-fat dairy products",
                        "Trans fats and fried foods"
                    ],
                    "meal_planning_tips": [
                        "Plan meals around colorful vegetables and fruits",
                        "Include plant-based proteins in your diet",
                        "Choose whole grains over refined carbohydrates",
                        "Practice portion control and mindful eating",
                        "Stay hydrated with water and green tea"
                    ]
                },
                "lifestyle_changes": {
                    "exercise_recommendations": [
                        "Aim for 150 minutes of moderate aerobic activity weekly",
                        "Include strength training exercises 2-3 times per week",
                        "Try yoga or Pilates for flexibility and stress relief",
                        "Take regular walks, especially after meals",
                        "Find physical activities you enjoy to maintain consistency"
                    ],
                    "stress_management": [
                        "Practice daily meditation or mindfulness",
                        "Engage in regular relaxation techniques",
                        "Maintain social connections and support networks",
                        "Consider counseling or therapy if needed"
                    ],
                    "sleep_hygiene": [
                        "Maintain a consistent sleep schedule",
                        "Aim for 7-9 hours of quality sleep nightly",
                        "Create a relaxing bedtime routine",
                        "Limit screen time before bed"
                    ]
                },
                "medical_care": {
                    "screening_schedule": "Discuss with your doctor about earlier or more frequent mammograms, possibly starting before age 50 and potentially including additional imaging like MRI",
                    "specialist_consultations": [
                        "Oncologist for risk assessment",
                        "Genetic counselor for hereditary risk evaluation",
                        "Nutritionist for personalized dietary guidance"
                    ],
                    "questions_for_doctor": [
                        "What is my specific level of breast cancer risk?",
                        "When should I start enhanced screening?",
                        "Should I consider genetic testing?",
                        "Are there any preventive medications I should consider?",
                        "What lifestyle changes would be most beneficial for me?"
                    ]
                },
                "monitoring_and_tracking": [
                    "Perform monthly breast self-examinations",
                    "Track any changes in breast tissue or symptoms",
                    "Monitor your weight and BMI",
                    "Keep a health journal of lifestyle changes",
                    "Track family health history updates"
                ],
                "support_resources": [
                    "American Cancer Society (www.cancer.org)",
                    "National Breast Cancer Foundation",
                    "BRCA Foundation for genetic mutation support",
                    "Local breast cancer support groups",
                    "Online communities like BreastCancer.org"
                ]
            }
        else:
            return {
                "refined_analysis": "Your current risk assessment indicates a lower risk of breast cancer. However, maintaining healthy lifestyle choices and regular screening remains important for continued breast health.",
                "immediate_actions": [
                    "Continue with routine preventive care and screening",
                    "Maintain your current healthy lifestyle choices",
                    "Stay informed about breast health and self-examination",
                    "Keep your healthcare provider updated on any changes"
                ],
                "diet_recommendations": {
                    "foods_to_include": [
                        "Variety of fruits and vegetables daily",
                        "Whole grains and fiber-rich foods",
                        "Lean proteins including fish and plant-based options",
                        "Healthy fats from nuts, seeds, and olive oil",
                        "Calcium-rich foods for overall health",
                        "Antioxidant-rich foods like berries and green tea"
                    ],
                    "foods_to_avoid": [
                        "Excessive processed foods",
                        "High amounts of saturated fats",
                        "Excessive alcohol consumption",
                        "Too much added sugar",
                        "Highly processed snack foods"
                    ],
                    "meal_planning_tips": [
                        "Follow a balanced, varied diet",
                        "Practice moderation in all food choices",
                        "Include 5-7 servings of fruits and vegetables daily",
                        "Choose whole foods over processed options",
                        "Maintain a healthy weight through balanced nutrition"
                    ]
                },
                "lifestyle_changes": {
                    "exercise_recommendations": [
                        "Maintain regular physical activity",
                        "Include both cardio and strength training",
                        "Find activities you enjoy for long-term consistency",
                        "Aim for at least 30 minutes of activity most days",
                        "Include flexibility and balance exercises"
                    ],
                    "stress_management": [
                        "Continue healthy stress management practices",
                        "Maintain work-life balance",
                        "Engage in hobbies and social activities",
                        "Practice relaxation techniques as needed"
                    ],
                    "sleep_hygiene": [
                        "Maintain consistent sleep patterns",
                        "Ensure adequate sleep quality",
                        "Create a comfortable sleep environment",
                        "Address any sleep issues promptly"
                    ]
                },
                "medical_care": {
                    "screening_schedule": "Follow standard screening guidelines - typically annual mammograms starting at age 50, or earlier if recommended by your healthcare provider",
                    "specialist_consultations": [
                        "Regular primary care visits",
                        "Gynecologist for women's health",
                        "Any specialists as recommended by your doctor"
                    ],
                    "questions_for_doctor": [
                        "When should I start routine mammograms?",
                        "How often should I have breast exams?",
                        "Are there any symptoms I should watch for?",
                        "Should my family history change my screening schedule?",
                        "What lifestyle factors are most important for my health?"
                    ]
                },
                "monitoring_and_tracking": [
                    "Perform monthly breast self-examinations",
                    "Attend regular check-ups and screenings",
                    "Monitor any changes in health status",
                    "Keep track of family health history",
                    "Maintain awareness of breast health"
                ],
                "support_resources": [
                    "American Cancer Society for general information",
                    "Your healthcare provider's patient resources",
                    "Women's health organizations",
                    "Local community health resources",
                    "Online educational resources about breast health"
                ]
            }
    
    def _get_fallback_analysis(self, detection_result):
        """Fallback analysis if AI service fails"""
        if "malignant" in detection_result.lower():
            return {
                "interpretation": "The analysis suggests findings that require immediate medical attention. Please remember that this is a screening tool and not a definitive diagnosis.",
                "immediate_next_steps": [
                    "Contact your healthcare provider immediately",
                    "Schedule an urgent medical consultation",
                    "Gather your medical history and any relevant documents",
                    "Arrange for emotional support during this time"
                ],
                "medical_follow_up": [
                    "Comprehensive clinical breast examination",
                    "Additional imaging studies as recommended",
                    "Possible tissue biopsy for definitive diagnosis",
                    "Consultation with a breast specialist or oncologist",
                    "Discussion of next steps based on findings"
                ],
                "emotional_support": [
                    "Remember that early detection leads to better outcomes",
                    "Seek support from family, friends, or counselors",
                    "Consider joining support groups",
                    "Practice stress-reduction techniques",
                    "Focus on taking care of your overall wellbeing"
                ],
                "preparation_for_medical_visit": [
                    "Prepare a list of symptoms and concerns",
                    "Gather family medical history information",
                    "List current medications and supplements",
                    "Prepare questions for your healthcare provider",
                    "Consider bringing a support person with you"
                ],
                "support_resources": [
                    "American Cancer Society Helpline",
                    "National Breast Cancer Foundation",
                    "Local hospital cancer support services",
                    "Online support communities",
                    "Mental health counseling services"
                ],
                "important_reminders": [
                    "This is a screening tool, not a final diagnosis",
                    "Many concerning findings turn out to be benign",
                    "Early detection and treatment have excellent outcomes",
                    "You are not alone in this journey"
                ]
            }
        else:
            return {
                "interpretation": "The analysis suggests findings that appear benign or normal. This is encouraging, but regular monitoring remains important.",
                "immediate_next_steps": [
                    "Continue with routine breast health monitoring",
                    "Maintain regular check-ups with your healthcare provider",
                    "Keep performing monthly self-examinations",
                    "Stay informed about breast health"
                ],
                "medical_follow_up": [
                    "Follow routine screening schedule",
                    "Discuss results with your healthcare provider",
                    "Continue regular clinical breast examinations",
                    "Report any new changes or symptoms promptly",
                    "Maintain up-to-date medical records"
                ],
                "emotional_support": [
                    "Feel reassured by these encouraging results",
                    "Continue prioritizing your overall health and wellbeing",
                    "Stay connected with your support network",
                    "Maintain positive health habits",
                    "Share good news with loved ones"
                ],
                "preparation_for_medical_visit": [
                    "Discuss these results during your next routine visit",
                    "Ask about appropriate screening intervals",
                    "Review your risk factors with your provider",
                    "Update your family medical history if needed",
                    "Discuss any concerns or questions you have"
                ],
                "support_resources": [
                    "Breast health educational resources",
                    "American Cancer Society prevention information",
                    "Your healthcare provider's patient resources",
                    "Women's health organizations",
                    "Community wellness programs"
                ],
                "important_reminders": [
                    "Continue regular breast health monitoring",
                    "Encouraging results don't eliminate the need for routine care",
                    "Report any new symptoms promptly",
                    "Maintain healthy lifestyle choices"
                ]
            }