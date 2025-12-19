import os
import sys
import pickle
from flask import Flask, render_template, request, redirect, url_for, flash

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

print("Starting Flask app...")
print(f"Python version: {sys.version}")

try:
    import numpy as np
    print("NumPy imported successfully")
    
    # Load the risk prediction model
    logreg_model = pickle.load(open("models/best_model.pkl", "rb"))
    print("Model loaded successfully")
    
    # Try to import AI agent
    try:
        from ai_agent import AIAgent
        ai_agent = AIAgent()
        ai_enabled = True
        print("AI Agent loaded successfully")
    except Exception as e:
        print(f"AI Agent failed to load: {e}")
        ai_agent = None
        ai_enabled = False
        
except Exception as e:
    print(f"Error loading dependencies: {e}")
    sys.exit(1)

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/what-is-breast-cancer")
def what_is_breast_cancer():
    return render_template("what-is-breast-cancer.html")

@app.route("/types")
def types():
    return render_template("types.html")

@app.route("/stages")
def stages():
    return render_template("stages.html")

@app.route("/early-detection")
def early_detection():
    return render_template("early-detection.html")

@app.route("/treatment")
def treatment():
    return render_template("treatment.html")

@app.route("/choose")
def choose():
    return render_template("choose.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/predict_risk", methods=["POST"])
def predict_risk():
    try:
        # Get form data
        family_history = int(request.form["Family_History"])
        genetic_mutation = int(request.form["Genetic_Mutation"])
        bmi = float(request.form["BMI"])
        age = int(request.form["Age"])
        previous_biopsy = int(request.form["Previous_Breast_Biopsy"])

        # Prepare input data
        input_data = np.array([[family_history, genetic_mutation, bmi, age, previous_biopsy]])
        
        # Make prediction
        prediction = logreg_model.predict(input_data)[0]
        prediction_prob = logreg_model.predict_proba(input_data)[0]
        
        # Interpret result
        if prediction == 1:
            prediction_text = "⚠️ High Risk of Breast Cancer"
            risk_color = "#d9534f"
        else:
            prediction_text = "✅ Low Risk of Breast Cancer"
            risk_color = "#5cb85c"
        
        # Calculate confidence
        confidence = max(prediction_prob) * 100
        
        # Generate diet recommendations based on risk level
        if prediction == 1:  # High risk
            diet_plan = [
                "🥬 Include plenty of leafy greens (spinach, kale, arugula)",
                "🍓 Consume antioxidant-rich berries (blueberries, strawberries)",
                "🐟 Eat omega-3 rich fish (salmon, mackerel, sardines)",
                "🥜 Add nuts and seeds (walnuts, flaxseeds, chia seeds)",
                "🥕 Include colorful vegetables (carrots, bell peppers, broccoli)",
                "❌ Limit processed meats and refined sugars",
                "🚫 Reduce alcohol consumption",
                "💧 Stay well hydrated with water and green tea"
            ]
        else:  # Low risk
            diet_plan = [
                "🥗 Maintain a balanced diet with variety",
                "🍎 Include 5-7 servings of fruits and vegetables daily",
                "🌾 Choose whole grains over refined carbohydrates",
                "🥩 Opt for lean proteins (chicken, fish, legumes)",
                "🥛 Include calcium-rich foods (dairy, leafy greens)",
                "⚖️ Maintain a healthy weight through balanced nutrition",
                "🏃‍♀️ Combine good nutrition with regular exercise",
                "😊 Enjoy meals mindfully and maintain social connections"
            ]
        
        # Prepare result data
        result = {
            'risk_level': prediction_text,
            'risk_color': risk_color,
            'risk_score': round(confidence, 1),
            'confidence': f"{confidence:.1f}%",
            'recommendation': "Consult with a healthcare provider for personalized advice and screening recommendations.",
            'diet_plan': diet_plan,
            'input_data': {
                'Age': age,
                'Family_History': "Yes" if family_history == 1 else "No",
                'Genetic_Mutation': "Yes" if genetic_mutation == 1 else "No",
                'BMI': bmi,
                'Previous_Biopsy': "Yes" if previous_biopsy == 1 else "No"
            }
        }
        
        # Get AI recommendations if available
        ai_recommendations = None
        if ai_enabled and ai_agent:
            try:
                ai_recommendations = ai_agent.get_risk_recommendations(
                    prediction_text, 
                    confidence, 
                    result['input_data']
                )
                print("AI recommendations generated successfully")
            except Exception as e:
                print(f"AI recommendations failed: {e}")
                ai_recommendations = None
        
        return render_template("predict.html", 
                             result=result, 
                             prediction_text=prediction_text,
                             ai_recommendations=ai_recommendations)
    
    except Exception as e:
        flash(f"⚠️ Error in prediction: {str(e)}", "error")
        return redirect(url_for("predict"))

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/detect_cancer", methods=["POST"])
def detect_cancer():
    try:
        # Get form data for numeric detection
        mean_radius = float(request.form.get("mean_radius", 0))
        mean_texture = float(request.form.get("mean_texture", 0))
        mean_perimeter = float(request.form.get("mean_perimeter", 0))
        mean_area = float(request.form.get("mean_area", 0))
        mean_smoothness = float(request.form.get("mean_smoothness", 0))

        # Simple fallback logic for detection
        if mean_radius > 15 or mean_area > 800:
            detection_text = "⚠️ Concerning features detected - Please consult a healthcare provider"
        else:
            detection_text = "✅ Features appear normal - Continue regular monitoring"
        
        # Get AI analysis if available
        ai_analysis = None
        if ai_enabled and ai_agent:
            try:
                ai_analysis = ai_agent.get_detection_analysis(detection_text)
                print("AI analysis generated successfully")
            except Exception as e:
                print(f"AI analysis failed: {e}")
                ai_analysis = None
        
        return render_template("detect.html", 
                             detection_text=detection_text,
                             ai_analysis=ai_analysis)
    
    except Exception as e:
        flash(f"⚠️ Error in detection: {str(e)}", "error")
        return redirect(url_for("detect"))

@app.route("/diagnosis")
def diagnosis():
    return render_template("diagnosis.html")

if __name__ == "__main__":
    print("Starting Flask development server...")
    app.run(debug=True)