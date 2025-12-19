from flask import Flask, render_template, request, redirect, url_for, flash, send_filefrom flask import Flask, render_template, request, redirect, url_for, flash, send_fileimport os

import pickle

import numpy as npimport pickleimport pickle

import pandas as pd

from datetime import datetimeimport numpy as npfrom flask import Flask, render_template, request, redirect, url_for, flash, send_file

import os

import pandas as pdfrom datetime import datetime

# Flask app setup

app = Flask(__name__)from datetime import datetimefrom medical_report_generator import MedicalReportGenerator

app.config['UPLOAD_FOLDER'] = 'static/uploads'

app.config['SECRET_KEY'] = 'your_secret_key'import os



# Create necessary directoriesfrom medical_report_generator import MedicalReportGenerator# Flask app setup

if not os.path.exists('reports'):

    os.makedirs('reports')import cv2app = Flask(__name__)

if not os.path.exists('static/uploads'):

    os.makedirs('static/uploads')from tensorflow.keras.preprocessing.image import load_img, img_to_arrayapp.config['UPLOAD_FOLDER'] = 'static/uploads'



# Load modelsfrom tensorflow.keras.models import load_modelapp.config['SECRET_KEY'] = 'your_secret_key'

try:

    with open("models/best_model.pkl", "rb") as f:

        logreg_model = pickle.load(f)

    print("Risk prediction model loaded successfully")# Flask app setup# Create reports directory if it doesn't exist

except Exception as e:

    print(f"Error loading risk model: {e}")app = Flask(__name__)if not os.path.exists('reports'):

    logreg_model = None

app.config['UPLOAD_FOLDER'] = 'static/uploads'    os.makedirs('reports')

# AI Agent Integration (with fallback)

def get_ai_recommendations(prediction_data):app.config['SECRET_KEY'] = 'your_secret_key'

    """

    This function integrates with AI agents for comprehensive recommendations# For simplified testing, let's create mock data instead of loading models

    Falls back to basic recommendations if AI agents are not available

    """# Create necessary directoriesprint("Starting simplified Flask app for PDF testing...")

    try:

        # Try to import AI agents (may fail due to dependency issues)if not os.path.exists('reports'):

        from ai_agent import AIAgent

        from gemini_service import GeminiService    os.makedirs('reports')# Initialize PDF report generator

        

        # Initialize AI agentif not os.path.exists('static/uploads'):try:

        ai_agent = AIAgent()

            os.makedirs('static/uploads')    pdf_generator = MedicalReportGenerator()

        # Get AI-powered recommendations

        recommendations = ai_agent.get_comprehensive_recommendations(prediction_data)    print("✅ PDF generator initialized successfully")

        

        return recommendations# Load modelsexcept Exception as e:

    except ImportError as e:

        print(f"AI Agent import error: {e}")logreg_model = pickle.load(open("models/best_model.pkl", "rb"))       # Risk prediction model    print(f"❌ PDF generator error: {e}")

        return get_fallback_recommendations(prediction_data)

    except Exception as e:    pdf_generator = None

        print(f"AI Agent error: {e}")

        return get_fallback_recommendations(prediction_data)try:



def get_fallback_recommendations(prediction_data):    cnn_model = load_model("model_cnn.h5")# Routes

    """

    Fallback recommendations when AI agents are not availableexcept:@app.route("/")

    """

    risk_level = prediction_data.get('risk_level', 'Low')    cnn_model = Nonedef home():

    

    if risk_level in ['High', 'Very High'] or prediction_data.get('prediction') == 'Malignant':    print("Warning: CNN model not found. Image classification will not work.")    return render_template("index.html")

        return {

            'immediate_actions': [

                'Consult with an oncologist immediately',

                'Schedule comprehensive breast imaging',# Initialize PDF report generator@app.route("/about")

                'Consider genetic counseling',

                'Seek second opinion if needed'report_generator = MedicalReportGenerator()def about():

            ],

            'lifestyle_changes': [    return render_template("about.html")

                'Maintain a healthy diet rich in fruits and vegetables',

                'Exercise regularly (at least 30 minutes daily)',@app.route("/")

                'Limit alcohol consumption',

                'Avoid smoking and tobacco products',def home():@app.route("/what-is-breast-cancer")

                'Maintain healthy weight'

            ],    return render_template("index.html")def what_is_breast_cancer():

            'medical_follow_up': [

                'Regular mammograms every 6-12 months',    return render_template("what-is-breast-cancer.html")

                'Clinical breast examinations every 3-6 months',

                'Consider MRI screening if recommended',@app.route("/about")

                'Blood tests and tumor markers if applicable'

            ],def about():@app.route("/types")

            'emotional_support': [

                'Join support groups',    return render_template("about.html")def types():

                'Consider counseling services',

                'Connect with breast cancer survivors',    return render_template("types.html")

                'Maintain strong family and friend support'

            ],@app.route("/detection")

            'support_resources': [

                'American Cancer Society: www.cancer.org',def detection():@app.route("/stages")

                'National Cancer Institute: www.cancer.gov',

                'Breast Cancer Research Foundation',    return render_template("detection.html")def stages():

                'Local cancer support centers'

            ]    return render_template("stages.html")

        }

    else:@app.route("/risk-assessment")

        return {

            'immediate_actions': [def risk_assessment():@app.route("/early-detection")

                'Continue regular health checkups',

                'Perform monthly self-examinations',    return render_template("risk_assessment.html")def early_detection():

                'Maintain current screening schedule',

                'Stay informed about breast health'    return render_template("early-detection.html")

            ],

            'lifestyle_changes': [# AI Agent Integration (requires external dependencies)

                'Maintain a balanced, nutritious diet',

                'Stay physically active',def get_ai_recommendations(prediction_data):@app.route("/treatment")

                'Limit alcohol consumption',

                'Avoid smoking',    """def treatment():

                'Manage stress effectively'

            ],    This function would integrate with AI agents for comprehensive recommendations    return render_template("treatment.html")

            'medical_follow_up': [

                'Annual mammograms as per guidelines',    Currently returns basic recommendations due to dependency issues

                'Regular check-ups with primary care physician',

                'Follow recommended screening protocols',    """@app.route("/choose")

                'Report any changes to healthcare provider'

            ],    try:def choose():

            'emotional_support': [

                'Stay connected with healthcare team',        # Import AI agents (may fail due to dependency issues)    return render_template("choose.html")

                'Maintain healthy relationships',

                'Practice stress management techniques',        from ai_agent import AIAgent

                'Stay positive and informed'

            ],        from gemini_service import GeminiService@app.route("/predict")

            'support_resources': [

                'Preventive care guidelines',        def predict():

                'Breast health education resources',

                'Wellness programs',        # Initialize AI agent    return render_template("predict.html")

                'Health maintenance apps'

            ]        ai_agent = AIAgent()

        }

        @app.route("/predict_risk", methods=["POST"])

@app.route("/")

def home():        # Get AI-powered recommendationsdef predict_risk():

    return render_template("index.html")

        recommendations = ai_agent.get_comprehensive_recommendations(prediction_data)    try:

@app.route("/about")

def about():                # Get form data

    return render_template("about.html")

        return recommendations        family_history = int(request.form["Family_History"])

@app.route("/detection")

def detection():    except ImportError as e:        genetic_mutation = int(request.form["Genetic_Mutation"])

    return render_template("detection.html")

        print(f"AI Agent import error: {e}")        bmi = float(request.form["BMI"])

@app.route("/risk-assessment")

def risk_assessment():        return get_fallback_recommendations(prediction_data)        age = int(request.form["Age"])

    return render_template("risk_assessment.html")

    except Exception as e:        previous_biopsy = int(request.form["Previous_Breast_Biopsy"])

@app.route("/predict", methods=["POST"])

def predict():        print(f"AI Agent error: {e}")

    if request.method == "POST":

        try:        return get_fallback_recommendations(prediction_data)        # Prepare input data

            if logreg_model is None:

                flash("Prediction model not available", "error")        input_data = np.array([[family_history, genetic_mutation, bmi, age, previous_biopsy]])

                return redirect(url_for("risk_assessment"))

                def get_fallback_recommendations(prediction_data):        

            # Get form data

            radius_mean = float(request.form["radius_mean"])    """        # Make prediction

            texture_mean = float(request.form["texture_mean"])

            perimeter_mean = float(request.form["perimeter_mean"])    Fallback recommendations when AI agents are not available        prediction = logreg_model.predict(input_data)[0]

            area_mean = float(request.form["area_mean"])

            smoothness_mean = float(request.form["smoothness_mean"])    """        prediction_prob = logreg_model.predict_proba(input_data)[0]

            compactness_mean = float(request.form["compactness_mean"])

            concavity_mean = float(request.form["concavity_mean"])    if prediction_data.get('risk_level') in ['High', 'Very High']:        

            concave_points_mean = float(request.form["concave_points_mean"])

            symmetry_mean = float(request.form["symmetry_mean"])        return {        # Interpret result

            fractal_dimension_mean = float(request.form["fractal_dimension_mean"])

            radius_se = float(request.form["radius_se"])            'immediate_actions': [        if prediction == 1:

            texture_se = float(request.form["texture_se"])

            perimeter_se = float(request.form["perimeter_se"])                'Consult with an oncologist immediately',            prediction_text = "⚠️ High Risk of Breast Cancer"

            area_se = float(request.form["area_se"])

            smoothness_se = float(request.form["smoothness_se"])                'Schedule comprehensive breast imaging',            risk_color = "#d9534f"

            compactness_se = float(request.form["compactness_se"])

            concavity_se = float(request.form["concavity_se"])                'Consider genetic counseling'        else:

            concave_points_se = float(request.form["concave_points_se"])

            symmetry_se = float(request.form["symmetry_se"])            ],            prediction_text = "✅ Low Risk of Breast Cancer"

            fractal_dimension_se = float(request.form["fractal_dimension_se"])

            radius_worst = float(request.form["radius_worst"])            'lifestyle_changes': [            risk_color = "#5cb85c"

            texture_worst = float(request.form["texture_worst"])

            perimeter_worst = float(request.form["perimeter_worst"])                'Maintain a healthy diet rich in fruits and vegetables',        

            area_worst = float(request.form["area_worst"])

            smoothness_worst = float(request.form["smoothness_worst"])                'Exercise regularly (at least 30 minutes daily)',        # Calculate confidence

            compactness_worst = float(request.form["compactness_worst"])

            concavity_worst = float(request.form["concavity_worst"])                'Limit alcohol consumption',        confidence = max(prediction_prob) * 100

            concave_points_worst = float(request.form["concave_points_worst"])

            symmetry_worst = float(request.form["symmetry_worst"])                'Avoid smoking'        

            fractal_dimension_worst = float(request.form["fractal_dimension_worst"])

            ],        # Generate diet recommendations based on risk level

            # Create feature array

            features = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,            'medical_follow_up': [        if prediction == 1:  # High risk

                                compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,

                                fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,                'Regular mammograms every 6-12 months',            diet_plan = [

                                smoothness_se, compactness_se, concavity_se, concave_points_se,

                                symmetry_se, fractal_dimension_se, radius_worst, texture_worst,                'Clinical breast examinations',                "🥬 Include plenty of leafy greens (spinach, kale, arugula)",

                                perimeter_worst, area_worst, smoothness_worst, compactness_worst,

                                concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]])                'Consider MRI screening if recommended'                "🍓 Consume antioxidant-rich berries (blueberries, strawberries)",



            # Make prediction            ]                "🐟 Eat omega-3 rich fish (salmon, mackerel, sardines)",

            prediction = logreg_model.predict(features)[0]

            probability = logreg_model.predict_proba(features)[0]        }                "🥜 Add nuts and seeds (walnuts, flaxseeds, chia seeds)",

            

            result = "Malignant" if prediction == 1 else "Benign"    else:                "🥕 Include colorful vegetables (carrots, bell peppers, broccoli)",

            confidence = max(probability) * 100

                    return {                "❌ Limit processed meats and refined sugars",

            # Calculate risk level for AI recommendations

            risk_level = 'High' if prediction == 1 else 'Low'            'immediate_actions': [                "🚫 Reduce alcohol consumption",

            

            # Get AI-powered recommendations                'Continue regular health checkups',                "💧 Stay well hydrated with water and green tea"

            prediction_data = {

                'result': result,                'Perform monthly self-examinations',            ]

                'confidence': confidence,

                'risk_level': risk_level,                'Maintain current screening schedule'        else:  # Low risk

                'prediction': result,

                'features': features.tolist()[0]            ],            diet_plan = [

            }

                        'lifestyle_changes': [                "🥗 Maintain a balanced diet with variety",

            ai_recommendations = get_ai_recommendations(prediction_data)

                            'Maintain a balanced diet',                "🍎 Include 5-7 servings of fruits and vegetables daily",

            # Patient data for potential report generation

            patient_data = {                'Stay physically active',                "🌾 Choose whole grains over refined carbohydrates",

                'name': request.form.get('patient_name', 'Patient'),

                'age': request.form.get('patient_age', 'N/A'),                'Limit alcohol and avoid smoking'                "🥩 Opt for lean proteins (chicken, fish, legumes)",

                'date': datetime.now().strftime('%Y-%m-%d'),

                'prediction': result,            ],                "🥛 Include calcium-rich foods (dairy, leafy greens)",

                'confidence': f"{confidence:.2f}%",

                'malignant_prob': f"{probability[1]*100:.2f}%",            'medical_follow_up': [                "⚖️ Maintain a healthy weight through balanced nutrition",

                'benign_prob': f"{probability[0]*100:.2f}%"

            }                'Annual mammograms as per guidelines',                "🏃‍♀️ Combine good nutrition with regular exercise",

            

            # Try to generate PDF report if medical report generator is available                'Regular check-ups with primary care physician'                "😊 Enjoy meals mindfully and maintain social connections"

            try:

                from medical_report_generator import MedicalReportGenerator            ]            ]

                report_generator = MedicalReportGenerator()

                        }        

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                filename = f"risk_assessment_report_{timestamp}.pdf"        # Prepare result data

                filepath = os.path.join("reports", filename)

                @app.route("/predict", methods=["POST"])        result = {

                report_generator.generate_risk_assessment_report(

                    patient_data, ai_recommendations, filepathdef predict():            'risk_level': prediction_text,

                )

                    if request.method == "POST":            'risk_color': risk_color,

                return render_template("result.html", 

                                     prediction=result,         try:            'risk_score': round(confidence, 1),

                                     confidence=f"{confidence:.2f}%",

                                     malignant_prob=f"{probability[1]*100:.2f}%",            # Get form data            'confidence': f"{confidence:.1f}%",

                                     benign_prob=f"{probability[0]*100:.2f}%",

                                     ai_recommendations=ai_recommendations,            radius_mean = float(request.form["radius_mean"])            'recommendation': "Consult with a healthcare provider for personalized advice and screening recommendations.",

                                     report_filename=filename)

            except ImportError:            texture_mean = float(request.form["texture_mean"])            'diet_plan': diet_plan,

                print("Medical report generator not available")

                return render_template("result.html",             perimeter_mean = float(request.form["perimeter_mean"])            'input_data': {

                                     prediction=result, 

                                     confidence=f"{confidence:.2f}%",            area_mean = float(request.form["area_mean"])                'Age': age,

                                     malignant_prob=f"{probability[1]*100:.2f}%",

                                     benign_prob=f"{probability[0]*100:.2f}%",            smoothness_mean = float(request.form["smoothness_mean"])                'Family_History': "Yes" if family_history == 1 else "No",

                                     ai_recommendations=ai_recommendations)

            except Exception as report_error:            compactness_mean = float(request.form["compactness_mean"])                'Genetic_Mutation': "Yes" if genetic_mutation == 1 else "No",

                print(f"Report generation error: {report_error}")

                return render_template("result.html",             concavity_mean = float(request.form["concavity_mean"])                'BMI': bmi,

                                     prediction=result, 

                                     confidence=f"{confidence:.2f}%",            concave_points_mean = float(request.form["concave_points_mean"])                'Previous_Biopsy': "Yes" if previous_biopsy == 1 else "No"

                                     malignant_prob=f"{probability[1]*100:.2f}%",

                                     benign_prob=f"{probability[0]*100:.2f}%",            symmetry_mean = float(request.form["symmetry_mean"])            }

                                     ai_recommendations=ai_recommendations)

                        fractal_dimension_mean = float(request.form["fractal_dimension_mean"])        }

        except Exception as e:

            flash(f"Error in prediction: {str(e)}", "error")            radius_se = float(request.form["radius_se"])        

            return redirect(url_for("risk_assessment"))

            texture_se = float(request.form["texture_se"])        # Get AI recommendations

@app.route('/download-report/<filename>')

def download_report(filename):            perimeter_se = float(request.form["perimeter_se"])        ai_recommendations = None

    try:

        return send_file(f'reports/{filename}', as_attachment=True)            area_se = float(request.form["area_se"])        try:

    except Exception as e:

        flash(f"Error downloading report: {str(e)}", "error")            smoothness_se = float(request.form["smoothness_se"])            if ai_agent:

        return redirect(url_for("home"))

            compactness_se = float(request.form["compactness_se"])                print("Attempting to get AI recommendations...")

if __name__ == "__main__":

    app.run(debug=True)            concavity_se = float(request.form["concavity_se"])                ai_recommendations = ai_agent.get_risk_recommendations(

            concave_points_se = float(request.form["concave_points_se"])                    prediction_text, 

            symmetry_se = float(request.form["symmetry_se"])                    confidence, 

            fractal_dimension_se = float(request.form["fractal_dimension_se"])                    result['input_data']

            radius_worst = float(request.form["radius_worst"])                )

            texture_worst = float(request.form["texture_worst"])                print(f"AI recommendations received: {ai_recommendations is not None}")

            perimeter_worst = float(request.form["perimeter_worst"])            else:

            area_worst = float(request.form["area_worst"])                print("AI agent not available")

            smoothness_worst = float(request.form["smoothness_worst"])        except Exception as e:

            compactness_worst = float(request.form["compactness_worst"])            print(f"AI recommendations failed: {e}")

            concavity_worst = float(request.form["concavity_worst"])            ai_recommendations = None

            concave_points_worst = float(request.form["concave_points_worst"])        

            symmetry_worst = float(request.form["symmetry_worst"])        # If AI failed, ensure we still get fallback recommendations

            fractal_dimension_worst = float(request.form["fractal_dimension_worst"])        if ai_recommendations is None and ai_agent:

            print("Using fallback recommendations...")

            # Create feature array            ai_recommendations = ai_agent._get_fallback_recommendations(prediction_text)

            features = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,        

                                compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,        return render_template("predict.html", 

                                fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,                             result=result, 

                                smoothness_se, compactness_se, concavity_se, concave_points_se,                             prediction_text=prediction_text,

                                symmetry_se, fractal_dimension_se, radius_worst, texture_worst,                             ai_recommendations=ai_recommendations)

                                perimeter_worst, area_worst, smoothness_worst, compactness_worst,    

                                concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]])    except Exception as e:

        flash(f"⚠️ Error in prediction: {str(e)}", "error")

            # Make prediction        return redirect(url_for("predict"))

            prediction = logreg_model.predict(features)[0]

            probability = logreg_model.predict_proba(features)[0]@app.route("/detect")

            def detect():

            result = "Malignant" if prediction == 1 else "Benign"    return render_template("detect.html")

            confidence = max(probability) * 100

            @app.route("/detect_cancer", methods=["POST"])

            # Calculate risk level for AI recommendationsdef detect_cancer():

            risk_level = 'High' if prediction == 1 else 'Low'    try:

                    # Get form data for numeric detection

            # Get AI-powered recommendations        mean_radius = float(request.form.get("mean_radius", 0))

            prediction_data = {        mean_texture = float(request.form.get("mean_texture", 0))

                'result': result,        mean_perimeter = float(request.form.get("mean_perimeter", 0))

                'confidence': confidence,        mean_area = float(request.form.get("mean_area", 0))

                'risk_level': risk_level,        mean_smoothness = float(request.form.get("mean_smoothness", 0))

                'features': features.tolist()[0]

            }        # Prepare input data for detection model

                    detection_input = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]])

            ai_recommendations = get_ai_recommendations(prediction_data)        

                    # Make prediction using detection model (if available)

            # Generate PDF report with AI recommendations        try:

            patient_data = {            detection_model = pickle.load(open("models/model.pkl", "rb"))

                'name': request.form.get('patient_name', 'Patient'),            detection_prediction = detection_model.predict(detection_input)[0]

                'age': request.form.get('patient_age', 'N/A'),            

                'date': datetime.now().strftime('%Y-%m-%d'),            if detection_prediction == 1:

                'prediction': result,                detection_text = "⚠️ Malignant - Requires immediate medical attention"

                'confidence': f"{confidence:.2f}%",            else:

                'malignant_prob': f"{probability[1]*100:.2f}%",                detection_text = "✅ Benign - Appears to be non-cancerous"

                'benign_prob': f"{probability[0]*100:.2f}%"        except:

            }            # Fallback simple logic if model not available

                        if mean_radius > 15 or mean_area > 800:

            # Generate report with AI recommendations                detection_text = "⚠️ Concerning features detected - Please consult a healthcare provider"

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")            else:

            filename = f"risk_assessment_report_{timestamp}.pdf"                detection_text = "✅ Features appear normal - Continue regular monitoring"

            filepath = os.path.join("reports", filename)        

                    # Get AI analysis using detection agent

            try:        try:

                report_generator.generate_risk_assessment_report(            if detection_agent:

                    patient_data, ai_recommendations, filepath                ai_analysis = detection_agent.get_detection_analysis(detection_text)

                )            else:

                                ai_analysis = None

                return render_template("result.html",         except Exception as e:

                                     prediction=result,             print(f"Detection AI analysis failed: {e}")

                                     confidence=f"{confidence:.2f}%",            ai_analysis = None

                                     malignant_prob=f"{probability[1]*100:.2f}%",        

                                     benign_prob=f"{probability[0]*100:.2f}%",        return render_template("detect.html", 

                                     ai_recommendations=ai_recommendations,                             detection_text=detection_text,

                                     report_filename=filename)                             ai_analysis=ai_analysis)

            except Exception as report_error:    

                print(f"Report generation error: {report_error}")    except Exception as e:

                return render_template("result.html",         flash(f"⚠️ Error in detection: {str(e)}", "error")

                                     prediction=result,         return redirect(url_for("detect"))

                                     confidence=f"{confidence:.2f}%",

                                     malignant_prob=f"{probability[1]*100:.2f}%",@app.route("/diagnosis")

                                     benign_prob=f"{probability[0]*100:.2f}%",def diagnosis():

                                     ai_recommendations=ai_recommendations)    return render_template("diagnosis.html")

            

        except Exception as e:if __name__ == "__main__":

            flash(f"Error in prediction: {str(e)}", "error")    app.run(debug=True)
            return redirect(url_for("risk_assessment"))

@app.route("/image-predict", methods=["POST"])
def image_predict():
    if cnn_model is None:
        flash("Image classification model not available", "error")
        return redirect(url_for("detection"))
        
    if request.method == "POST":
        try:
            # Handle image upload
            if 'image' not in request.files:
                flash('No image file uploaded', 'error')
                return redirect(url_for("detection"))
            
            file = request.files['image']
            if file.filename == '':
                flash('No image selected', 'error')
                return redirect(url_for("detection"))
            
            # Save uploaded image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"upload_{timestamp}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess image for CNN
            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            # Make prediction
            prediction = cnn_model.predict(img_array)[0][0]
            
            # Convert to classification
            if prediction > 0.5:
                result = "Malignant"
                confidence = prediction * 100
            else:
                result = "Benign"  
                confidence = (1 - prediction) * 100
            
            # Get AI recommendations for detection
            detection_data = {
                'result': result,
                'confidence': confidence,
                'image_path': filepath
            }
            
            ai_recommendations = get_ai_recommendations(detection_data)
            
            # Generate detection report
            patient_data = {
                'name': request.form.get('patient_name', 'Patient'),
                'age': request.form.get('patient_age', 'N/A'),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'prediction': result,
                'confidence': f"{confidence:.2f}%",
                'image_filename': filename
            }
            
            report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"detection_report_{report_timestamp}.pdf"
            report_filepath = os.path.join("reports", report_filename)
            
            try:
                report_generator.generate_detection_report(
                    patient_data, ai_recommendations, report_filepath
                )
                
                return render_template("detection_result.html",
                                     prediction=result,
                                     confidence=f"{confidence:.2f}%",
                                     image_filename=filename,
                                     ai_recommendations=ai_recommendations,
                                     report_filename=report_filename)
            except Exception as report_error:
                print(f"Report generation error: {report_error}")
                return render_template("detection_result.html",
                                     prediction=result,
                                     confidence=f"{confidence:.2f}%",
                                     image_filename=filename,
                                     ai_recommendations=ai_recommendations)
            
        except Exception as e:
            flash(f"Error in image prediction: {str(e)}", "error")
            return redirect(url_for("detection"))

@app.route('/download-report/<filename>')
def download_report(filename):
    try:
        return send_file(f'reports/{filename}', as_attachment=True)
    except Exception as e:
        flash(f"Error downloading report: {str(e)}", "error")
        return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)