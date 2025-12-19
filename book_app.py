from flask import Flask, request, jsonify
from hospital_service import HospitalService
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
hs = HospitalService()

@app.route("/book", methods=["POST"])
def book():
    data = request.json   # USER ka data

    booking_payload = {
        "name": data.get("name"),
        "email": data.get("email"),
        "preferred_date": data.get("preferred_date"),
        "detection_text": data.get("detection_text")
    }

    result = hs.book_appointment(booking_payload)
    return jsonify(result)


if __name__ == "__main__":
    # If you want to run the Flask server, uncomment below:
    # app.run(debug=True)

    # Quick test as requested (STEP 5):
    print("Running quick local booking test using HospitalService...")

    test_payload = {
        "name": os.getenv('TEST_NAME', 'Test User'),
        "email": os.getenv('TEST_EMAIL', 'yourgmail@gmail.com'),
        "preferred_date": os.getenv('TEST_DATE', 'Tomorrow'),
        "detection_text": "Test result"
    }

    result = hs.book_appointment(test_payload)
    print('Test booking result:')
    print(result)

    # Helpful note for user
    print('\nIf email_sent is False, check:\n- .env has SMTP settings (EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT, EMAIL_FROM)\n- If using Gmail, use an app password and set EMAIL_USER and EMAIL_PASS\n- Run a local SMTP debug server for testing: python local_smtp_debug_server.py (listens on 127.0.0.1:1025 by default)')