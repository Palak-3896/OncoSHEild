import os
import pickle
import numpy as np
from werkzeug.utils import secure_filename

# TensorFlow imports for image classification
try:
    import tensorflow as tf
    from PIL import Image
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow loaded for image classification")
except ImportError as e:
    print(f"❌ TensorFlow not available: {e}")
    TENSORFLOW_AVAILABLE = False

def generate_diet_intro(risk_level):
    """Generate appropriate diet introduction based on risk level"""
    if risk_level == "LOW":
        return gemini_service.model.generate_content(
            "Generate a reassuring but informative message about the importance of following dietary recommendations for cancer prevention, specifically for someone with LOW cancer risk. Keep it concise and encouraging."
        ).text if gemini_service else "Maintaining a healthy diet is important for cancer prevention, even with your currently low risk level."
    elif risk_level == "MODERATE":
        return gemini_service.model.generate_content(
            "Generate a message about the importance of dietary changes for someone with MODERATE cancer risk. Emphasize how diet can help manage risk factors."
        ).text if gemini_service else "Following these dietary guidelines can help manage your risk factors and promote overall health."
    else:  # HIGH risk
        return gemini_service.model.generate_content(
            "Generate an urgent but supportive message about the critical importance of dietary changes for someone with HIGH cancer risk. Emphasize how diet works alongside medical care."
        ).text if gemini_service else "Following these dietary guidelines is crucial as part of your comprehensive risk management plan."
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from datetime import datetime
from medical_report_generator import MedicalReportGenerator
from gemini_service import GeminiService

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'your_secret_key'

# Create reports directory if it doesn't exist
if not os.path.exists('reports'):
    os.makedirs('reports')

# Create uploads directory for image uploads
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

# Allowed file extensions for image upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

print("Starting Flask app with PDF generation...")

# Initialize PDF report generator
try:
    pdf_generator = MedicalReportGenerator()
    print("✅ PDF generator initialized successfully")
except Exception as e:
    print(f"❌ PDF generator error: {e}")
    pdf_generator = None

# Initialize Gemini AI service
try:
    gemini_service = GeminiService()
    print("✅ Gemini AI service initialized successfully")
except Exception as e:
    print(f"❌ Gemini AI service error: {e}")
    gemini_service = None

# Initialize Gemini AI service
try:
    gemini_service = GeminiService()
    print("✅ Gemini AI service initialized successfully")
except Exception as e:
    print(f"❌ Gemini AI service error: {e}")
    gemini_service = None

# Load CNN model for image classification
cnn_model = None
if TENSORFLOW_AVAILABLE:
    try:
        cnn_model = tf.keras.models.load_model("model_cnn.h5")
        print("✅ CNN image classification model loaded")
        print(f"   Input shape: {cnn_model.input_shape}")
        print(f"   Output shape: {cnn_model.output_shape}")
    except Exception as e:
        print(f"❌ CNN model error: {e}")
else:
    print("⚠️ CNN model cannot be loaded (TensorFlow not available)")

# Define class labels for CNN model
CLASS_LABELS = ['benign', 'malignant', 'normal']

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_likely_ultrasound_image(image_path):
    """
    Check if the uploaded image appears to be a medical ultrasound image.
    Ultrasound images typically have:
    - Grayscale or predominantly dark appearance
    - Lower average brightness
    - Medical imaging characteristics
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Calculate average brightness
        avg_brightness = np.mean(img_array)
        
        # Calculate color variance (ultrasound images tend to be more grayscale)
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        color_variance = np.var([np.mean(r), np.mean(g), np.mean(b)])
        
        # Check image dimensions (ultrasound images have certain aspect ratios)
        width, height = img.size
        aspect_ratio = width / height
        
        # Heuristic checks for ultrasound characteristics
        # Ultrasound images typically: darker (avg_brightness < 150), more grayscale (low color_variance)
        is_dark_enough = avg_brightness < 180  # Not too bright like normal photos
        is_grayscale_like = color_variance < 100  # Channels are similar (grayscale-ish)
        has_reasonable_aspect = 0.5 < aspect_ratio < 2.0  # Reasonable medical image aspect
        
        print(f"Image validation - Brightness: {avg_brightness:.1f}, Color variance: {color_variance:.1f}, Aspect: {aspect_ratio:.2f}")
        
        # At least 2 out of 3 criteria should match for ultrasound
        matches = sum([is_dark_enough, is_grayscale_like, has_reasonable_aspect])
        
        return matches >= 2
    except Exception as e:
        print(f"Error validating ultrasound image: {e}")
        return True  # If validation fails, allow the image through

def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocess ultrasound image for CNN model (trained on 128x128 ultrasound images)"""
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        # Resize to model's expected input size (same as training)
        img = img.resize(target_size)
        # Convert to numpy array
        img_array = np.array(img)
        # Normalize pixel values to [0, 1] (same as training preprocessing)
        img_array = img_array / 255.0
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def generate_comprehensive_recommendations(patient_data, is_high_risk):
    """Generate AI-powered comprehensive recommendations using Gemini"""
    
    # Calculate risk score for more granular analysis
    risk_score = (patient_data['Family_History'] * 30 + 
                 patient_data['Genetic_Mutation'] * 40 + 
                 max(0, patient_data['BMI'] - 25) * 2 + 
                 max(0, patient_data['Age'] - 45) * 1 + 
                 patient_data['Previous_Breast_Biopsy'] * 20)
    
    # Determine risk level
    if risk_score > 50:
        risk_level = "HIGH"
    elif risk_score > 30:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"
    
    # Use Gemini AI to generate personalized recommendations
    if gemini_service:
        try:
            print(f"🤖 Calling Gemini AI for {risk_level} risk level (score: {risk_score}%)")
            ai_recommendations = gemini_service.generate_risk_recommendations(
                risk_level=risk_level,
                risk_score=risk_score,
                input_data=patient_data
            )
            if ai_recommendations:
                print(f"✅ AI recommendations generated for {risk_level} risk level")
                print(f"📊 AI response keys: {list(ai_recommendations.keys())}")
                return ai_recommendations
            else:
                print("⚠️ AI service returned None, using fallback")
        except Exception as e:
            print(f"❌ Error generating AI recommendations: {e}")
    else:
        print("⚠️ Gemini service not available, using fallback")
    
    # Fallback to simplified recommendations if AI fails
    if is_high_risk:
        return {
            'refined_analysis': f"Based on your risk profile (Age: {patient_data['Age']}, BMI: {patient_data['BMI']}, Family History: {'Yes' if patient_data['Family_History'] else 'No'}, Genetic Mutation: {'Yes' if patient_data['Genetic_Mutation'] else 'No'}), you have an elevated risk for breast cancer. Please consult with healthcare professionals for personalized guidance.",
            'immediate_actions': [
                "Schedule consultation with oncologist within 2-4 weeks",
                "Consider genetic counseling if mutations are present",
                "Request enhanced screening protocols from healthcare provider",
                "Begin monthly breast self-examinations immediately"
            ],
            'diet_recommendations': {
                'introduction': "Following a strict dietary regimen is particularly important given your current risk level. These recommendations are designed to support your overall health and risk management.",
                'foods_to_include': [
                    "Dark leafy greens rich in antioxidants",
                    "Omega-3 fatty fish 2-3 times weekly",
                    "Colorful fruits and vegetables",
                    "Whole grains and lean proteins"
                ],
                'foods_to_avoid': [
                    "Processed and red meats",
                    "Excessive alcohol consumption",
                    "Refined sugars and processed foods",
                    "Trans fats and hydrogenated oils"
                ]
            },
            'lifestyle_changes': {
                'exercise_recommendations': [
                    "Exercise 150+ minutes moderate activity weekly",
                    "Include both cardio and strength training",
                    "Daily walking for at least 30 minutes"
                ],
                'stress_management': [
                    "Practice meditation daily",
                    "Deep breathing exercises",
                    "Engage in relaxing activities"
                ],
                'sleep_hygiene': [
                    "Prioritize 7-9 hours quality sleep nightly",
                    "Maintain consistent bedtime routine",
                    "Limit screen time before bed"
                ]
            },
            'medical_care': {
                'screening_schedule': "Enhanced screening recommended - discuss with your healthcare provider",
                'specialist_consultations': [
                    "Oncologist for risk assessment",
                    "Genetic counselor if mutations present",
                    "Breast specialist for screening protocols"
                ],
                'questions_for_doctor': [
                    "What are my specific risk factors?",
                    "Should I consider preventive medications?",
                    "What screening schedule is appropriate?"
                ]
            },
            'monitoring_and_tracking': [
                "Monthly breast self-examinations",
                "Track family history changes",
                "Monitor weight and BMI",
                "Keep symptom diary"
            ],
            'support_resources': [
                "National Cancer Institute (cancer.gov)",
                "American Cancer Society (cancer.org)",
                "Susan G. Komen Foundation (komen.org)",
                "Local cancer support groups"
            ]
        }
    else:
        return {
            'refined_analysis': f"Your current risk assessment indicates lower breast cancer risk. Continue preventive measures to maintain good health.",
            'immediate_actions': [
                "Continue routine screening as age-appropriate",
                "Maintain regular healthcare check-ups",
                "Perform monthly breast self-examinations",
                "Stay informed about breast health"
            ],
            'diet_recommendations': {
                'introduction': generate_diet_intro(risk_level),
                'foods_to_include': [
                    "Variety of fruits and vegetables daily",
                    "Whole grains and lean proteins",
                    "Healthy fats from nuts and seeds",
                    "Low-fat dairy or plant alternatives"
                ],
                'foods_to_avoid': [
                    "Limit processed foods",
                    "Reduce alcohol consumption",
                    "Minimize refined sugars",
                    "Avoid excessive saturated fats"
                ]
            },
            'lifestyle_changes': {
                'exercise_recommendations': [
                    "Regular physical activity 30+ minutes daily",
                    "Include strength training twice weekly",
                    "Yoga or stretching for flexibility"
                ],
                'stress_management': [
                    "Practice relaxation techniques",
                    "Engage in enjoyable hobbies",
                    "Maintain social connections"
                ],
                'sleep_hygiene': [
                    "Maintain regular sleep schedule",
                    "Create comfortable sleep environment",
                    "Practice good sleep hygiene"
                ]
            },
            'medical_care': {
                'screening_schedule': "Standard screening as recommended by healthcare provider",
                'specialist_consultations': [
                    "Primary care physician for routine care",
                    "Gynecologist for annual exams",
                    "Discuss screening timing with doctor"
                ],
                'questions_for_doctor': [
                    "When should I start regular mammography?",
                    "How often should I do self-examinations?",
                    "What lifestyle factors should I focus on?"
                ]
            },
            'monitoring_and_tracking': [
                "Monthly breast self-examinations",
                "Track family history updates",
                "Monitor overall health metrics",
                "Keep up with routine appointments"
            ],
            'support_resources': [
                "American Cancer Society prevention information",
                "National Cancer Institute educational resources",
                "Local wellness and health groups",
                "Healthcare provider materials"
            ]
        }

def generate_detection_analysis(test_data, is_malignant):
    """Generate AI-powered comprehensive detection analysis using Gemini"""
    
    # Determine detection result
    detection_result = "Malignant (Cancerous)" if is_malignant else "Benign (Non-Cancerous)"
    
    # Use Gemini AI to generate personalized detection analysis
    if gemini_service:
        try:
            print(f"🤖 Calling Gemini AI for detection analysis: {detection_result}")
            ai_analysis = gemini_service.generate_detection_analysis(
                detection_result=detection_result,
                image_features=test_data
            )
            if ai_analysis:
                print(f"✅ AI detection analysis generated for {detection_result}")
                print(f"📊 AI analysis keys: {list(ai_analysis.keys())}")
                return ai_analysis
            else:
                print("⚠️ AI service returned None for detection analysis, using fallback")
        except Exception as e:
            print(f"❌ Error generating AI detection analysis: {e}")
    else:
        print("⚠️ Gemini service not available for detection analysis, using fallback")
    
    # Fallback to simplified analysis if AI fails
    if is_malignant:
        return {
            'interpretation': f"The tumor characteristics (radius: {test_data['radius_mean']:.1f}, texture: {test_data['texture_mean']:.1f}, area: {test_data['area_mean']:.1f}) show features consistent with malignant tissue. These measurements exceed typical benign ranges and warrant immediate medical evaluation.",
            'immediate_next_steps': [
                "Contact oncologist or healthcare provider immediately",
                "Schedule confirmatory biopsy if not already completed",
                "Gather all medical records and imaging for consultation",
                "Prepare list of questions for medical team"
            ],
            'medical_follow_up': [
                "Histopathological examination for definitive diagnosis",
                "Staging studies (CT, MRI, PET scan if indicated)",
                "Tumor marker testing and receptor status analysis",
                "Multidisciplinary team consultation"
            ],
            'emotional_support': [
                "Build strong support network of family and friends",
                "Consider joining cancer support groups",
                "Practice mindfulness and relaxation techniques",
                "Seek professional counseling if needed"
            ],
            'support_resources': [
                "American Cancer Society (cancer.org)",
                "National Cancer Institute (cancer.gov)",
                "Susan G. Komen Foundation (komen.org)",
                "CancerCare (cancercare.org)"
            ],
            'important_reminders': [
                "This requires additional testing for final diagnosis",
                "Early detection and treatment improve outcomes",
                "You are not alone - support systems are available",
                "Treatment options have improved significantly"
            ]
        }
    else:
        return {
            'interpretation': f"The tumor characteristics (radius: {test_data['radius_mean']:.1f}, texture: {test_data['texture_mean']:.1f}, area: {test_data['area_mean']:.1f}) are within ranges typically associated with benign tissue. However, continued monitoring and healthy lifestyle practices remain important.",
            'immediate_next_steps': [
                "Continue regular breast health monitoring",
                "Discuss results with your healthcare provider",
                "Maintain scheduled screening appointments",
                "Monitor for any changes in breast tissue"
            ],
            'medical_follow_up': [
                "Regular clinical breast examinations as scheduled",
                "Mammography according to age and risk guidelines",
                "Annual comprehensive health assessments",
                "Monitor for any new breast changes or symptoms"
            ],
            'emotional_support': [
                "Feel reassured by these positive results",
                "Celebrate this good news with loved ones",
                "Use this as motivation for continued health",
                "Continue stress management practices"
            ],
            'support_resources': [
                "Breast health educational websites",
                "Reliable medical information sources",
                "Your healthcare provider's patient portal",
                "Breast health awareness organizations"
            ],
            'important_reminders': [
                "Continue regular self-examinations monthly",
                "Don't skip scheduled mammograms and check-ups",
                "Report any new changes to your healthcare provider",
                "Maintain healthy lifestyle choices"
            ]
        }

def generate_image_detection_analysis(detection_result, confidence, image_path=None):
    """Generate AI-powered analysis for image-based detection"""
    
    # Use Gemini AI to generate personalized image analysis
    if gemini_service:
        try:
            print(f"🤖 Calling Gemini AI for image detection analysis: {detection_result}")
            
            # Create detailed prompt for image detection
            prompt = f"""
            You are a medical AI assistant providing guidance for breast cancer ultrasound image detection.
            
            Detection Result: {detection_result}
            Confidence Level: {confidence:.1f}%
            
            Provide comprehensive analysis in JSON format with these exact keys:
            {{
                "interpretation": "Clear explanation of what the detection result means",
                "confidence_assessment": "Assessment of detection confidence and any limitations",
                "immediate_next_steps": ["List of 4-5 immediate actions the patient should take"],
                "medical_consultation": {{
                    "urgency": "Description of urgency level (Low/Medium/High)",
                    "specialist": "Type of specialist to consult",
                    "what_to_expect": "What to expect during medical consultation"
                }},
                "additional_testing": ["List of 3-4 additional tests that may be recommended"],
                "lifestyle_recommendations": ["List of 4-5 lifestyle and health recommendations"],
                "emotional_support": ["List of 4-5 emotional coping strategies"],
                "support_resources": ["List of 4-5 support organizations and resources"],
                "important_notes": ["List of 3-4 key reminders about AI detection limitations"]
            }}
            
            Be compassionate, informative, and emphasize that this is AI-assisted detection requiring medical confirmation.
            """
            
            response = gemini_service.model.generate_content(prompt)
            ai_analysis = gemini_service._parse_json_response(response.text)
            
            if ai_analysis:
                print(f"✅ AI image detection analysis generated")
                return ai_analysis
            else:
                print("⚠️ AI service returned None for image detection, using fallback")
        except Exception as e:
            print(f"❌ Error generating AI image detection analysis: {e}")
    else:
        print("⚠️ Gemini service not available for image detection, using fallback")
    
    # Fallback analysis based on detection result
    if "malignant" in detection_result.lower():
        return {
            'interpretation': f"The ultrasound image analysis suggests features consistent with malignant tissue. This AI-assisted detection has a confidence level of {confidence:.1f}%. Please note that this is a preliminary assessment and requires professional medical evaluation.",
            'confidence_assessment': f"The model confidence is {confidence:.1f}%. AI models can assist but should never replace professional medical diagnosis. Further testing is essential.",
            'immediate_next_steps': [
                "Contact your healthcare provider or oncologist immediately",
                "Schedule a professional radiological review of the ultrasound",
                "Request a biopsy for definitive diagnosis",
                "Gather all your medical records and previous imaging",
                "Don't panic - early detection improves treatment outcomes"
            ],
            'medical_consultation': {
                'urgency': "High - Schedule consultation within 1-2 weeks",
                'specialist': "Oncologist, Radiologist, or Breast Specialist",
                'what_to_expect': "Expect detailed imaging review, possible additional scans, discussion of biopsy procedures, and comprehensive risk assessment"
            },
            'additional_testing': [
                "Professional ultrasound review by radiologist",
                "Mammography or additional imaging",
                "Core needle biopsy for tissue analysis",
                "Blood tests including tumor markers"
            ],
            'lifestyle_recommendations': [
                "Maintain a healthy diet rich in fruits, vegetables, and whole grains",
                "Engage in regular physical activity (150 minutes per week)",
                "Limit alcohol consumption and avoid smoking",
                "Maintain healthy body weight",
                "Get adequate sleep (7-9 hours nightly)"
            ],
            'emotional_support': [
                "Reach out to family and friends for support",
                "Consider joining a breast cancer support group",
                "Practice stress-reduction techniques like meditation",
                "Seek professional counseling if feeling overwhelmed",
                "Remember that early detection significantly improves outcomes"
            ],
            'support_resources': [
                "American Cancer Society: www.cancer.org (1-800-227-2345)",
                "National Cancer Institute: www.cancer.gov",
                "Susan G. Komen Foundation: www.komen.org",
                "Breastcancer.org for comprehensive information",
                "Local hospital cancer support services"
            ],
            'important_notes': [
                "This is an AI-assisted screening tool, not a diagnostic device",
                "Only a qualified medical professional can provide definitive diagnosis",
                "False positives can occur - professional confirmation is essential",
                "Do not delay seeking professional medical evaluation"
            ]
        }
    elif "benign" in detection_result.lower():
        return {
            'interpretation': f"The ultrasound image analysis suggests features consistent with benign (non-cancerous) tissue. The AI model's confidence level is {confidence:.1f}%. While this is encouraging, professional medical confirmation is still important.",
            'confidence_assessment': f"The model shows {confidence:.1f}% confidence in benign classification. However, only medical professionals can provide definitive diagnosis through comprehensive evaluation.",
            'immediate_next_steps': [
                "Schedule follow-up with your healthcare provider",
                "Discuss the AI result with your doctor",
                "Continue regular screening schedule",
                "Maintain breast self-examination routine",
                "Stay vigilant for any changes"
            ],
            'medical_consultation': {
                'urgency': "Moderate - Schedule routine follow-up within 4-6 weeks",
                'specialist': "Primary care physician or Breast Specialist",
                'what_to_expect': "Professional review of imaging, discussion of findings, and recommendations for ongoing monitoring"
            },
            'additional_testing': [
                "Professional ultrasound interpretation",
                "Follow-up imaging as recommended",
                "Clinical breast examination",
                "Routine mammography per age guidelines"
            ],
            'lifestyle_recommendations': [
                "Continue healthy diet with plenty of fruits and vegetables",
                "Maintain regular exercise routine",
                "Limit alcohol and avoid tobacco",
                "Manage stress through relaxation techniques",
                "Keep up with preventive health measures"
            ],
            'emotional_support': [
                "Stay informed about breast health",
                "Maintain positive outlook while being health-conscious",
                "Share concerns with healthcare provider",
                "Connect with wellness communities",
                "Focus on preventive health measures"
            ],
            'support_resources': [
                "American Cancer Society prevention resources",
                "National Cancer Institute screening guidelines",
                "Breastcancer.org for health information",
                "Local women's health centers",
                "Healthcare provider educational materials"
            ],
            'important_notes': [
                "Benign findings still require medical verification",
                "Continue regular screening and self-examinations",
                "Report any new symptoms or changes to your doctor",
                "AI tools assist but don't replace medical judgment"
            ]
        }
    else:  # normal
        return {
            'interpretation': f"The ultrasound image appears to show normal breast tissue. The AI confidence level is {confidence:.1f}%. This is a positive indicator, but routine medical monitoring remains important.",
            'confidence_assessment': f"Normal tissue detected with {confidence:.1f}% confidence. Professional confirmation and continued screening are recommended.",
            'immediate_next_steps': [
                "Continue regular breast health monitoring",
                "Maintain scheduled screening appointments",
                "Perform monthly breast self-examinations",
                "Discuss results with healthcare provider at next visit",
                "Stay informed about breast health"
            ],
            'medical_consultation': {
                'urgency': "Low - Routine follow-up as scheduled",
                'specialist': "Primary care physician for routine screening",
                'what_to_expect': "Continue age-appropriate screening schedule and routine health monitoring"
            },
            'additional_testing': [
                "Continue routine mammography per guidelines",
                "Regular clinical breast examinations",
                "Self-monitoring and awareness",
                "Follow age-appropriate screening protocols"
            ],
            'lifestyle_recommendations': [
                "Maintain healthy balanced diet",
                "Regular physical activity",
                "Healthy weight management",
                "Limit alcohol consumption",
                "Avoid tobacco products"
            ],
            'emotional_support': [
                "Celebrate good health while staying vigilant",
                "Stay educated about breast health",
                "Encourage others to get screened",
                "Maintain healthy lifestyle habits",
                "Focus on preventive care"
            ],
            'support_resources': [
                "Preventive health information resources",
                "Breast health education materials",
                "Wellness and healthy living communities",
                "Healthcare provider guidance",
                "Cancer prevention organizations"
            ],
            'important_notes': [
                "Normal results are encouraging but not a guarantee",
                "Continue regular screening and monitoring",
                "Report any new symptoms immediately",
                "AI detection is a screening aid, not a substitute for medical care"
            ]
        }

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/choose", methods=["GET", "POST"])
def choose():
    if request.method == "POST":
        choice = request.form.get("choice")
        if choice == "predict":
            return redirect(url_for('predict'))
        elif choice == "detect":
            return redirect(url_for('detect'))
        elif choice == "image":
            return redirect(url_for('detect_image_page'))
    return render_template("choose.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/detect_image")
def detect_image_page():
    """Image detection page"""
    if not TENSORFLOW_AVAILABLE or cnn_model is None:
        flash("⚠️ Image detection is currently unavailable. Please try again later.", "error")
        return redirect(url_for('choose'))
    return render_template("detect_image.html")

# About section routes
@app.route("/about/what-is-breast-cancer")
def what_is_breast_cancer():
    return render_template("what-is-breast-cancer.html")

@app.route("/about/early-detection")
def early_detection():
    return render_template("early-detection.html")

@app.route("/about/diagnosis")
def diagnosis():
    return render_template("diagnosis.html")

@app.route("/about/stages")
def stages():
    return render_template("stages.html")

@app.route("/about/types")
def types():
    return render_template("types.html")

@app.route("/about/treatment")
def treatment():
    return render_template("treatment.html")

@app.route("/predict_risk", methods=["POST"])
def predict_risk():
    try:
        # Get form data
        patient_data = {
            'Family_History': int(request.form.get("Family_History", 0)),
            'Genetic_Mutation': int(request.form.get("Genetic_Mutation", 0)),
            'BMI': float(request.form.get("BMI", 25.0)),
            'Age': int(request.form.get("Age", 30)),
            'Previous_Breast_Biopsy': int(request.form.get("Previous_Breast_Biopsy", 0))
        }
        
        # Mock prediction logic for testing
        risk_score = (patient_data['Family_History'] * 30 + 
                     patient_data['Genetic_Mutation'] * 40 + 
                     max(0, patient_data['BMI'] - 25) * 2 + 
                     max(0, patient_data['Age'] - 45) * 1 + 
                     patient_data['Previous_Breast_Biopsy'] * 20)
        
        result = "⚠ High Risk of Developing Cancer" if risk_score > 30 else "✅ Low Risk of Developing Cancer"
        is_high_risk = risk_score > 30
        
        # Generate comprehensive recommendations
        ai_recommendations = generate_comprehensive_recommendations(patient_data, is_high_risk)
        
        # Generate PDF report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"risk_assessment_report_{timestamp}.pdf"
        pdf_path = os.path.join("reports", pdf_filename)
        
        try:
            if pdf_generator:
                pdf_generator.generate_prediction_report(patient_data, result, pdf_path, ai_recommendations)
                pdf_generated = True
                download_link = f"/download_report/{pdf_filename}"
                print(f"✅ PDF report generated: {pdf_filename}")
            else:
                pdf_generated = False
                download_link = None
        except Exception as pdf_error:
            print(f"PDF generation error: {pdf_error}")
            pdf_generated = False
            download_link = None
        
        return render_template("predict.html", 
                             prediction_text=result,
                             patient_data=patient_data,
                             ai_recommendations=ai_recommendations,
                             pdf_generated=pdf_generated,
                             download_link=download_link,
                             report_filename=pdf_filename)
    except Exception as e:
        error_message = f"⚠ Error: {str(e)}"
        return render_template("predict.html", prediction_text=error_message)

@app.route("/detect_cancer", methods=["POST"])
def detect_cancer():
    try:
        # Get form data
        test_data = {
            'radius_mean': float(request.form.get("radius_mean", 10.0)),
            'texture_mean': float(request.form.get("texture_mean", 15.0)),
            'perimeter_mean': float(request.form.get("perimeter_mean", 80.0)),
            'area_mean': float(request.form.get("area_mean", 500.0)),
            'smoothness_mean': float(request.form.get("smoothness_mean", 0.1))
        }
        
        # Mock detection logic for testing
        malignancy_score = (test_data['radius_mean'] * 2 + 
                           test_data['texture_mean'] * 1.5 + 
                           test_data['perimeter_mean'] * 0.5 +
                           test_data['area_mean'] * 0.01 +
                           test_data['smoothness_mean'] * 100)
        
        detection_text = "⚠ Malignant (Cancerous)" if malignancy_score > 100 else "✅ Benign (Non-Cancerous)"
        is_malignant = malignancy_score > 100
        
        # Generate comprehensive analysis
        ai_analysis = generate_detection_analysis(test_data, is_malignant)
        
        # Generate PDF report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"detection_report_{timestamp}.pdf"
        pdf_path = os.path.join("reports", pdf_filename)
        
        try:
            if pdf_generator:
                pdf_generator.generate_detection_report(test_data, detection_text, pdf_path, ai_analysis)
                pdf_generated = True
                download_link = f"/download_report/{pdf_filename}"
                print(f"✅ PDF report generated: {pdf_filename}")
            else:
                pdf_generated = False
                download_link = None
        except Exception as pdf_error:
            print(f"PDF generation error: {pdf_error}")
            pdf_generated = False
            download_link = None
            
        return render_template("detect.html", 
                             detection_text=detection_text,
                             test_data=test_data,
                             ai_analysis=ai_analysis,
                             pdf_generated=pdf_generated,
                             download_link=download_link,
                             report_filename=pdf_filename)
    except Exception as e:
        error_message = f"⚠ Error: {str(e)}"
        return render_template("detect.html", detection_text=error_message)

@app.route("/detect_image_submit", methods=["POST"])
def detect_image_submit():
    """Handle ultrasound image upload and detection"""
    try:
        # Check if CNN model is available
        if cnn_model is None:
            flash("⚠️ Image detection model not available", "error")
            return redirect(url_for('detect_image_page'))
        
        # Check if file was uploaded
        if 'image' not in request.files:
            flash("No image file provided", "error")
            return redirect(url_for('detect_image_page'))
        
        file = request.files['image']
        
        if file.filename == '':
            flash("No file selected", "error")
            return redirect(url_for('detect_image_page'))
        
        if not allowed_file(file.filename):
            flash(f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}", "error")
            return redirect(url_for('detect_image_page'))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"📸 Image saved: {filepath}")
        
        # Validate if image appears to be an ultrasound image
        if not is_likely_ultrasound_image(filepath):
            flash("⚠️ Warning: This doesn't appear to be an ultrasound image. Please upload a breast ultrasound image for accurate detection. The model is specifically trained on ultrasound imaging data.", "warning")
            # Still allow processing but with warning
        
        # Preprocess image (trained on 128x128 ultrasound images)
        img_array = preprocess_image(filepath, target_size=(128, 128))
        
        if img_array is None:
            flash("Error processing image. Please ensure you upload a valid ultrasound image.", "error")
            return redirect(url_for('detect_image_page'))
        
        # Make prediction
        predictions = cnn_model.predict(img_array)[0]
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASS_LABELS[predicted_class_idx]
        confidence = predictions[predicted_class_idx] * 100
        
        print(f"🔬 Prediction: {predicted_class} ({confidence:.1f}% confidence)")
        print(f"   All probabilities - Benign: {predictions[0]*100:.1f}%, Malignant: {predictions[1]*100:.1f}%, Normal: {predictions[2]*100:.1f}%")
        
        # Check confidence level and provide additional warning if low
        if confidence < 60:
            flash(f"⚠️ Low confidence detection ({confidence:.1f}%). This may indicate a non-ultrasound image or poor image quality. Please ensure you're uploading a clear breast ultrasound image for reliable results.", "warning")
        
        # Format detection result
        if predicted_class == 'malignant':
            detection_text = "⚠ Malignant (Potentially Cancerous)"
            result_color = "#d9534f"
        elif predicted_class == 'benign':
            detection_text = "✅ Benign (Non-Cancerous)"
            result_color = "#5cb85c"
        else:  # normal
            detection_text = "✅ Normal Tissue"
            result_color = "#5cb85c"
        
        # Generate AI analysis
        ai_analysis = generate_image_detection_analysis(predicted_class, confidence, filepath)
        
        # Generate PDF report
        timestamp_pdf = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"image_detection_report_{timestamp_pdf}.pdf"
        pdf_path = os.path.join("reports", pdf_filename)
        
        try:
            if pdf_generator:
                # Create test data dict for PDF
                test_data = {
                    'image_path': f"uploads/{filename}",
                    'predicted_class': predicted_class,
                    'confidence': f"{confidence:.1f}%",
                    'all_probabilities': {
                        'benign': f"{predictions[0]*100:.1f}%",
                        'malignant': f"{predictions[1]*100:.1f}%",
                        'normal': f"{predictions[2]*100:.1f}%"
                    }
                }
                pdf_generator.generate_detection_report(test_data, detection_text, pdf_path, ai_analysis, image_path=filepath)
                pdf_generated = True
                download_link = f"/download_report/{pdf_filename}"
                print(f"✅ PDF report generated: {pdf_filename}")
            else:
                pdf_generated = False
                download_link = None
        except Exception as pdf_error:
            print(f"PDF generation error: {pdf_error}")
            pdf_generated = False
            download_link = None
        
        # Render result page
        return render_template("detect_image.html", 
                             detection_text=detection_text,
                             result_color=result_color,
                             confidence=f"{confidence:.1f}%",
                             predicted_class=predicted_class,
                             image_filename=filename,
                             all_probabilities={
                                 'Benign': f"{predictions[0]*100:.1f}%",
                                 'Malignant': f"{predictions[1]*100:.1f}%",
                                 'Normal': f"{predictions[2]*100:.1f}%"
                             },
                             ai_analysis=ai_analysis,
                             pdf_generated=pdf_generated,
                             download_link=download_link,
                             report_filename=pdf_filename)
    
    except Exception as e:
        print(f"❌ Image detection error: {e}")
        import traceback
        traceback.print_exc()
        flash(f"Error in detection: {str(e)}", "error")
        return redirect(url_for('detect_image_page'))

@app.route("/download_report/<filename>")
def download_report(filename):
    """Download generated PDF report"""
    try:
        report_path = os.path.join("reports", filename)
        if os.path.exists(report_path):
            return send_file(report_path, as_attachment=True, download_name=filename)
        else:
            flash("⚠ Report file not found!", "error")
            return redirect(url_for("home"))
    except Exception as e:
        flash(f"⚠ Error downloading report: {str(e)}", "error")
        return redirect(url_for("home"))

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏥 Breast Cancer Detection System")
    print("="*60)
    print(f"✓ Risk Prediction: Enabled")
    print(f"✓ Numeric Detection: Enabled")
    print(f"✓ Image Detection: {'Enabled' if cnn_model else 'Disabled'}")
    print(f"✓ AI Analysis: {'Enabled' if gemini_service else 'Disabled'}")
    print(f"✓ PDF Reports: {'Enabled' if pdf_generator else 'Disabled'}")
    print("="*60 + "\n")
    print("🚀 Starting Flask app...")
    app.run(debug=True, port=5000)