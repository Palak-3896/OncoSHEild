from flask import Flask, request, render_template, jsonifyfrom flask import Flask, request, render_template, send_file, jsonifyfrom flask import Flask, request, render_template, send_file, jsonify

import pickle

import numpy as npimport pickleimport pickle



app = Flask(__name__)import numpy as npimport numpy as np



# Load the trained modelimport pandas as pdimport pandas as pd

try:

    with open('models/best_model.pkl', 'rb') as f:from datetime import datetimefrom datetime import datetime

        model = pickle.load(f)

    print("Model loaded successfully")import osimport os

except Exception as e:

    print(f"Error loading model: {e}")

    model = None

app = Flask(__name__)app = Flask(__name__)

@app.route('/')

def home():

    return render_template('index.html')

# Load the trained model# Load the trained model

@app.route('/about')

def about():try:

    return render_template('about.html')

    with open('models/best_model.pkl', 'rb') as f:try:# Flask app setup

@app.route('/detection')

def detection():        model = pickle.load(f)

    return render_template('detection.html')

    print("Model loaded successfully")    with open('models/best_model.pkl', 'rb') as f:app = Flask(__name__)

@app.route('/risk-assessment')

def risk_assessment():except Exception as e:

    return render_template('risk_assessment.html')

    print(f"Error loading model: {e}")        model = pickle.load(f)app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/predict', methods=['POST'])

def predict():    model = None

    try:

        if model is None:    print("Model loaded successfully")app.config['SECRET_KEY'] = 'your_secret_key'

            return jsonify({'error': 'Model not loaded'})

        @app.route('/')

        # Get form data

        data = request.formdef home():except Exception as e:

        

        # Extract features    return render_template('index.html')

        features = [

            float(data.get('radius_mean', 0)),    print(f"Error loading model: {e}")# Create reports directory if it doesn't exist

            float(data.get('texture_mean', 0)),

            float(data.get('perimeter_mean', 0)),@app.route('/about')

            float(data.get('area_mean', 0)),

            float(data.get('smoothness_mean', 0)),def about():    model = Noneif not os.path.exists('reports'):

            float(data.get('compactness_mean', 0)),

            float(data.get('concavity_mean', 0)),    return render_template('about.html')

            float(data.get('concave_points_mean', 0)),

            float(data.get('symmetry_mean', 0)),    os.makedirs('reports')

            float(data.get('fractal_dimension_mean', 0)),

            float(data.get('radius_se', 0)),@app.route('/detection')

            float(data.get('texture_se', 0)),

            float(data.get('perimeter_se', 0)),def detection():@app.route('/')

            float(data.get('area_se', 0)),

            float(data.get('smoothness_se', 0)),    return render_template('detection.html')

            float(data.get('compactness_se', 0)),

            float(data.get('concavity_se', 0)),def home():# Load models

            float(data.get('concave_points_se', 0)),

            float(data.get('symmetry_se', 0)),@app.route('/risk-assessment')

            float(data.get('fractal_dimension_se', 0)),

            float(data.get('radius_worst', 0)),def risk_assessment():    return render_template('index.html')logreg_model = pickle.load(open("models/best_model.pkl", "rb"))       # Risk prediction model

            float(data.get('texture_worst', 0)),

            float(data.get('perimeter_worst', 0)),    return render_template('risk_assessment.html')

            float(data.get('area_worst', 0)),

            float(data.get('smoothness_worst', 0)),detection_model = pickle.load(open("models/model.pkl", "rb"))         # Numeric detection model

            float(data.get('compactness_worst', 0)),

            float(data.get('concavity_worst', 0)),@app.route('/predict', methods=['POST'])

            float(data.get('concave_points_worst', 0)),

            float(data.get('symmetry_worst', 0)),def predict():@app.route('/about')

            float(data.get('fractal_dimension_worst', 0))

        ]    try:

        

        # Make prediction        if model is None:def about():# Load CNN model (try to load, handle if not available)

        prediction = model.predict([features])[0]

        probability = model.predict_proba([features])[0]            return jsonify({'error': 'Model not loaded'})

        

        # Convert prediction to readable format            return render_template('about.html')try:

        result = 'Malignant' if prediction == 1 else 'Benign'

        confidence = max(probability) * 100        # Get form data

        

        return jsonify({        data = request.form    cnn_model = load_model("model_cnn.h5")

            'prediction': result,

            'confidence': f'{confidence:.2f}%',        

            'malignant_probability': f'{probability[1] * 100:.2f}%',

            'benign_probability': f'{probability[0] * 100:.2f}%'        # Extract features@app.route('/detection')except:

        })

                features = [

    except Exception as e:

        print(f"Prediction error: {e}")            float(data.get('radius_mean', 0)),def detection():    cnn_model = None

        return jsonify({'error': 'Prediction failed'})

            float(data.get('texture_mean', 0)),

@app.route('/risk-predict', methods=['POST'])

def risk_predict():            float(data.get('perimeter_mean', 0)),    return render_template('detection.html')    print("Warning: CNN model not found. Image classification will not work.")

    try:

        data = request.form            float(data.get('area_mean', 0)),

        

        # Basic risk assessment logic            float(data.get('smoothness_mean', 0)),

        risk_factors = []

        risk_score = 0            float(data.get('compactness_mean', 0)),

        

        # Age factor            float(data.get('concavity_mean', 0)),@app.route('/risk-assessment')# Initialize PDF report generator

        age = int(data.get('age', 0))

        if age > 50:            float(data.get('concave_points_mean', 0)),

            risk_score += 2

            risk_factors.append('Age over 50')            float(data.get('symmetry_mean', 0)),def risk_assessment():pdf_generator = MedicalReportGenerator()

        elif age > 40:

            risk_score += 1            float(data.get('fractal_dimension_mean', 0)),

            risk_factors.append('Age over 40')

                    float(data.get('radius_se', 0)),    return render_template('risk_assessment.html')

        # Family history

        if data.get('family_history') == 'yes':            float(data.get('texture_se', 0)),

            risk_score += 3

            risk_factors.append('Family history of breast cancer')            float(data.get('perimeter_se', 0)),# Routes

        

        # Personal history            float(data.get('area_se', 0)),

        if data.get('personal_history') == 'yes':

            risk_score += 4            float(data.get('smoothness_se', 0)),@app.route('/predict', methods=['POST'])@app.route("/")

            risk_factors.append('Personal history of breast cancer')

                    float(data.get('compactness_se', 0)),

        # Genetic mutations

        if data.get('genetic_mutations') == 'yes':            float(data.get('concavity_se', 0)),def predict():def home():

            risk_score += 5

            risk_factors.append('Genetic mutations (BRCA1/BRCA2)')            float(data.get('concave_points_se', 0)),

        

        # Lifestyle factors            float(data.get('symmetry_se', 0)),    try:    return render_template("index.html")

        if data.get('smoking') == 'yes':

            risk_score += 1            float(data.get('fractal_dimension_se', 0)),

            risk_factors.append('Smoking')

                    float(data.get('radius_worst', 0)),        if model is None:

        if data.get('alcohol') == 'frequent':

            risk_score += 1            float(data.get('texture_worst', 0)),

            risk_factors.append('Frequent alcohol consumption')

                    float(data.get('perimeter_worst', 0)),            return jsonify({'error': 'Model not loaded'})@app.route("/about")

        if data.get('physical_activity') == 'low':

            risk_score += 1            float(data.get('area_worst', 0)),

            risk_factors.append('Low physical activity')

                    float(data.get('smoothness_worst', 0)),        def about():

        # Determine risk level

        if risk_score >= 8:            float(data.get('compactness_worst', 0)),

            risk_level = 'Very High'

        elif risk_score >= 6:            float(data.get('concavity_worst', 0)),        # Get form data    return render_template("about.html")

            risk_level = 'High'

        elif risk_score >= 4:            float(data.get('concave_points_worst', 0)),

            risk_level = 'Moderate'

        elif risk_score >= 2:            float(data.get('symmetry_worst', 0)),        data = request.form

            risk_level = 'Low'

        else:            float(data.get('fractal_dimension_worst', 0))

            risk_level = 'Very Low'

                ]        # New About Section Routes

        # Generate basic recommendations

        recommendations = []        

        if risk_level in ['High', 'Very High']:

            recommendations = [        # Make prediction        # Extract features@app.route("/about/what-is-breast-cancer")

                'Consult with an oncologist immediately',

                'Consider genetic counseling',        prediction = model.predict([features])[0]

                'Schedule regular mammograms (every 6-12 months)',

                'Maintain a healthy lifestyle with regular exercise',        probability = model.predict_proba([features])[0]        features = [def what_is_breast_cancer():

                'Consider preventive measures after consulting with specialists'

            ]        

        elif risk_level == 'Moderate':

            recommendations = [        # Convert prediction to readable format            float(data.get('radius_mean', 0)),    return render_template("what-is-breast-cancer.html")

                'Schedule annual mammograms',

                'Consult with your primary care physician',        result = 'Malignant' if prediction == 1 else 'Benign'

                'Maintain a healthy diet and exercise routine',

                'Limit alcohol consumption',        confidence = max(probability) * 100            float(data.get('texture_mean', 0)),

                'Stay informed about breast health'

            ]        

        else:

            recommendations = [        return jsonify({            float(data.get('perimeter_mean', 0)),@app.route("/about/early-detection")

                'Continue regular health checkups',

                'Maintain a healthy lifestyle',            'prediction': result,

                'Perform regular self-examinations',

                'Stay physically active',            'confidence': f'{confidence:.2f}%',            float(data.get('area_mean', 0)),def early_detection():

                'Follow standard screening guidelines'

            ]            'malignant_probability': f'{probability[1] * 100:.2f}%',

        

        return jsonify({            'benign_probability': f'{probability[0] * 100:.2f}%'            float(data.get('smoothness_mean', 0)),    return render_template("early-detection.html")

            'risk_level': risk_level,

            'risk_score': risk_score,        })

            'risk_factors': risk_factors,

            'recommendations': recommendations                    float(data.get('compactness_mean', 0)),

        })

            except Exception as e:

    except Exception as e:

        print(f"Risk assessment error: {e}")        print(f"Prediction error: {e}")            float(data.get('concavity_mean', 0)),@app.route("/about/diagnosis")

        return jsonify({'error': 'Risk assessment failed'})

        return jsonify({'error': 'Prediction failed'})

if __name__ == '__main__':

    app.run(debug=True)            float(data.get('concave_points_mean', 0)),def diagnosis():

@app.route('/risk-predict', methods=['POST'])

def risk_predict():            float(data.get('symmetry_mean', 0)),    return render_template("diagnosis.html")

    try:

        data = request.form            float(data.get('fractal_dimension_mean', 0)),

        

        # Basic risk assessment logic            float(data.get('radius_se', 0)),@app.route("/about/stages")

        risk_factors = []

        risk_score = 0            float(data.get('texture_se', 0)),def stages():

        

        # Age factor            float(data.get('perimeter_se', 0)),    return render_template("stages.html")

        age = int(data.get('age', 0))

        if age > 50:            float(data.get('area_se', 0)),

            risk_score += 2

            risk_factors.append('Age over 50')            float(data.get('smoothness_se', 0)),@app.route("/about/types")

        elif age > 40:

            risk_score += 1            float(data.get('compactness_se', 0)),def types():

            risk_factors.append('Age over 40')

                    float(data.get('concavity_se', 0)),    return render_template("types.html")

        # Family history

        if data.get('family_history') == 'yes':            float(data.get('concave_points_se', 0)),

            risk_score += 3

            risk_factors.append('Family history of breast cancer')            float(data.get('symmetry_se', 0)),@app.route("/about/treatment")

        

        # Personal history            float(data.get('fractal_dimension_se', 0)),def treatment():

        if data.get('personal_history') == 'yes':

            risk_score += 4            float(data.get('radius_worst', 0)),    return render_template("treatment.html")

            risk_factors.append('Personal history of breast cancer')

                    float(data.get('texture_worst', 0)),

        # Genetic mutations

        if data.get('genetic_mutations') == 'yes':            float(data.get('perimeter_worst', 0)),@app.route("/choose", methods=["GET", "POST"])

            risk_score += 5

            risk_factors.append('Genetic mutations (BRCA1/BRCA2)')            float(data.get('area_worst', 0)),def choose():

        

        # Lifestyle factors            float(data.get('smoothness_worst', 0)),    if request.method == "POST":

        if data.get('smoking') == 'yes':

            risk_score += 1            float(data.get('compactness_worst', 0)),        choice = request.form.get("choice")

            risk_factors.append('Smoking')

                    float(data.get('concavity_worst', 0)),        if choice == "predict":

        if data.get('alcohol') == 'frequent':

            risk_score += 1            float(data.get('concave_points_worst', 0)),            return redirect(url_for("predict"))

            risk_factors.append('Frequent alcohol consumption')

                    float(data.get('symmetry_worst', 0)),        elif choice == "detect":

        if data.get('physical_activity') == 'low':

            risk_score += 1            float(data.get('fractal_dimension_worst', 0))            return redirect(url_for("detect"))

            risk_factors.append('Low physical activity')

                ]        elif choice == "image":

        # Determine risk level

        if risk_score >= 8:                    return redirect(url_for("detect_image"))

            risk_level = 'Very High'

        elif risk_score >= 6:        # Make prediction    return render_template("choose.html")

            risk_level = 'High'

        elif risk_score >= 4:        prediction = model.predict([features])[0]

            risk_level = 'Moderate'

        elif risk_score >= 2:        probability = model.predict_proba([features])[0]@app.route("/predict")

            risk_level = 'Low'

        else:        def predict():

            risk_level = 'Very Low'

                # Convert prediction to readable format    return render_template("predict.html")

        # Generate basic recommendations

        recommendations = []        result = 'Malignant' if prediction == 1 else 'Benign'

        if risk_level in ['High', 'Very High']:

            recommendations = [        confidence = max(probability) * 100@app.route("/detect")

                'Consult with an oncologist immediately',

                'Consider genetic counseling',        def detect():

                'Schedule regular mammograms (every 6-12 months)',

                'Maintain a healthy lifestyle with regular exercise',        return jsonify({    return render_template("detect.html")

                'Consider preventive measures after consulting with specialists'

            ]            'prediction': result,

        elif risk_level == 'Moderate':

            recommendations = [            'confidence': f'{confidence:.2f}%',@app.route("/detect_image")

                'Schedule annual mammograms',

                'Consult with your primary care physician',            'malignant_probability': f'{probability[1] * 100:.2f}%',def detect_image():

                'Maintain a healthy diet and exercise routine',

                'Limit alcohol consumption',            'benign_probability': f'{probability[0] * 100:.2f}%'    return render_template("detect_image.html")

                'Stay informed about breast health'

            ]        })

        else:

            recommendations = [        @app.route("/predict_risk", methods=["POST"])

                'Continue regular health checkups',

                'Maintain a healthy lifestyle',    except Exception as e:def predict_risk():

                'Perform regular self-examinations',

                'Stay physically active',        print(f"Prediction error: {e}")    try:

                'Follow standard screening guidelines'

            ]        return jsonify({'error': 'Prediction failed'})        # Get form data

        

        return jsonify({        patient_data = {

            'risk_level': risk_level,

            'risk_score': risk_score,@app.route('/risk-predict', methods=['POST'])            'Family_History': int(request.form["Family_History"]),

            'risk_factors': risk_factors,

            'recommendations': recommendationsdef risk_predict():            'Genetic_Mutation': int(request.form["Genetic_Mutation"]),

        })

            try:            'BMI': float(request.form["BMI"]),

    except Exception as e:

        print(f"Risk assessment error: {e}")        data = request.form            'Age': int(request.form["Age"]),

        return jsonify({'error': 'Risk assessment failed'})

                    'Previous_Breast_Biopsy': int(request.form["Previous_Breast_Biopsy"])

if __name__ == '__main__':

    app.run(debug=True)        # Basic risk assessment logic        }

        risk_factors = []        

        risk_score = 0        features = [

                    patient_data['Family_History'],

        # Age factor            patient_data['Genetic_Mutation'],

        age = int(data.get('age', 0))            patient_data['BMI'],

        if age > 50:            patient_data['Age'],

            risk_score += 2            patient_data['Previous_Breast_Biopsy']

            risk_factors.append('Age over 50')        ]

        elif age > 40:        

            risk_score += 1        features_array = np.array([features])

            risk_factors.append('Age over 40')        prediction = logreg_model.predict(features_array)[0]

                result = "⚠ High Risk of Developing Cancer" if prediction == 1 else "✅ Low Risk of Developing Cancer"

        # Family history        

        if data.get('family_history') == 'yes':        # Generate PDF report

            risk_score += 3        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            risk_factors.append('Family history of breast cancer')        pdf_filename = f"risk_assessment_report_{timestamp}.pdf"

                pdf_path = os.path.join("reports", pdf_filename)

        # Personal history        

        if data.get('personal_history') == 'yes':        try:

            risk_score += 4            pdf_generator.generate_prediction_report(patient_data, result, pdf_path)

            risk_factors.append('Personal history of breast cancer')            pdf_generated = True

                    download_link = f"/download_report/{pdf_filename}"

        # Genetic mutations        except Exception as pdf_error:

        if data.get('genetic_mutations') == 'yes':            print(f"PDF generation error: {pdf_error}")

            risk_score += 5            pdf_generated = False

            risk_factors.append('Genetic mutations (BRCA1/BRCA2)')            download_link = None

                

        # Lifestyle factors        return render_template("predict.html", 

        if data.get('smoking') == 'yes':                             prediction_text=result,

            risk_score += 1                             pdf_generated=pdf_generated,

            risk_factors.append('Smoking')                             download_link=download_link,

                                     report_filename=pdf_filename)

        if data.get('alcohol') == 'frequent':    except Exception as e:

            risk_score += 1        error_message = f"⚠ Error: {str(e)}"

            risk_factors.append('Frequent alcohol consumption')        return render_template("predict.html", prediction_text=error_message)

        

        if data.get('physical_activity') == 'low':@app.route("/detect_cancer", methods=["POST"])

            risk_score += 1def detect_cancer():

            risk_factors.append('Low physical activity')    detection_text = None

            try:

        # Determine risk level        # Get form data

        if risk_score >= 8:        test_data = {

            risk_level = 'Very High'            'radius_mean': float(request.form.get("radius_mean")),

        elif risk_score >= 6:            'texture_mean': float(request.form.get("texture_mean")),

            risk_level = 'High'            'perimeter_mean': float(request.form.get("perimeter_mean")),

        elif risk_score >= 4:            'area_mean': float(request.form.get("area_mean")),

            risk_level = 'Moderate'            'smoothness_mean': float(request.form.get("smoothness_mean"))

        elif risk_score >= 2:        }

            risk_level = 'Low'        

        else:        features = [

            risk_level = 'Very Low'            test_data['radius_mean'],

                    test_data['texture_mean'],

        # Generate basic recommendations            test_data['perimeter_mean'],

        recommendations = []            test_data['area_mean'],

        if risk_level in ['High', 'Very High']:            test_data['smoothness_mean']

            recommendations = [        ]

                'Consult with an oncologist immediately',        

                'Consider genetic counseling',        features_array = np.array([features])

                'Schedule regular mammograms (every 6-12 months)',        prediction = detection_model.predict(features_array)[0]

                'Maintain a healthy lifestyle with regular exercise',        detection_text = "⚠ Malignant (Cancerous)" if prediction == 1 else "✅ Benign (Non-Cancerous)"

                'Consider preventive measures after consulting with specialists'        

            ]        # Generate PDF report

        elif risk_level == 'Moderate':        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            recommendations = [        pdf_filename = f"detection_report_{timestamp}.pdf"

                'Schedule annual mammograms',        pdf_path = os.path.join("reports", pdf_filename)

                'Consult with your primary care physician',        

                'Maintain a healthy diet and exercise routine',        try:

                'Limit alcohol consumption',            pdf_generator.generate_detection_report(test_data, detection_text, pdf_path)

                'Stay informed about breast health'            pdf_generated = True

            ]            download_link = f"/download_report/{pdf_filename}"

        else:        except Exception as pdf_error:

            recommendations = [            print(f"PDF generation error: {pdf_error}")

                'Continue regular health checkups',            pdf_generated = False

                'Maintain a healthy lifestyle',            download_link = None

                'Perform regular self-examinations',            

                'Stay physically active',        return render_template("detect.html", 

                'Follow standard screening guidelines'                             detection_text=detection_text,

            ]                             pdf_generated=pdf_generated,

                                     download_link=download_link,

        return jsonify({                             report_filename=pdf_filename)

            'risk_level': risk_level,    except Exception as e:

            'risk_score': risk_score,        detection_text = f"⚠ Error: {str(e)}"

            'risk_factors': risk_factors,        return render_template("detect.html", detection_text=detection_text)

            'recommendations': recommendations

        })@app.route("/classify_image", methods=["POST"])

        def classify_image():

    except Exception as e:    if "image_file" not in request.files:

        print(f"Risk assessment error: {e}")        flash("⚠ No image uploaded!", "error")

        return jsonify({'error': 'Risk assessment failed'})        return redirect(url_for("detect_image"))



if __name__ == '__main__':    file = request.files["image_file"]

    app.run(debug=True)    if file.filename == "":
        flash("⚠ No selected file!", "error")
        return redirect(url_for("detect_image"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        image_cv = cv2.imread(filepath)
        if image_cv is None:
            raise ValueError("Invalid image format.")

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        # 1. Check grayscale level (color variance)
        b, g, r = cv2.split(image_cv)
        color_diff = np.std(b - g) + np.std(b - r)
        is_grayscale_like = color_diff < 15  # Allow small variation

        # 2. Check for texture (Laplacian)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_textured_like_ultrasound = laplacian_var > 10  # Adjusted threshold

        if not (is_grayscale_like and is_textured_like_ultrasound):
            raise ValueError("The image does not appear to be an ultrasound scan.")

    except Exception as e:
        flash(f"⚠ Rejected: {str(e)}", "error")
        os.remove(filepath)
        return redirect(url_for("detect_image"))

    try:
        # Prepare image for CNN
        img = load_img(filepath, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = cnn_model.predict(img_array)
        class_index = np.argmax(prediction[0])
        labels = ["benign", "malignant", "normal"]
        result = f"🧠 Predicted Class: {labels[class_index]}"
        return render_template("detect_image.html", prediction=result, image_path="/" + filepath)

    except Exception as e:
        flash(f"⚠ Error in image processing: {str(e)}", "error")
        os.remove(filepath)
        return redirect(url_for("detect_image"))

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



# Utility function
def secure_filename(filename):
    return filename.replace(" ", "_").replace("..", "").lower()

if __name__ == "__main__":
    app.run(debug=True)
