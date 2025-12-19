"""Minimal test runner for HospitalService.book_appointment

Run with:
    python test_hospital_service.py

It will print the returned dict. Make sure .env contains SMTP settings for email tests, or run a local SMTP debug server.
"""

from hospital_service import HospitalService
from dotenv import load_dotenv
import os

load_dotenv()

hs = HospitalService()

test_payload = {
    "name": os.getenv('TEST_NAME', 'Unit Test Patient'),
    "email": os.getenv('TEST_EMAIL', 'test-patient@example.com'),
    "preferred_date": os.getenv('TEST_DATE', '2025-12-20'),
    "detection_text": 'Malignant - unit test simulated',
}

print('Calling HospitalService.book_appointment with:')
print(test_payload)

result = hs.book_appointment(test_payload)

print('\nResult:')
print(result)

if not result.get('email_sent'):
    print('\nEmail was not sent. Troubleshooting tips:')
    print('- Ensure .env has EMAIL_SMTP_SERVER and EMAIL_SMTP_PORT set (for local tests use localhost:1025)')
    print("- If using Gmail, set EMAIL_USER and EMAIL_PASS to an app password and use SMTP_HOST smtp.gmail.com, port 587")
    print('- To inspect outgoing emails locally, run the included local SMTP debug server: python local_smtp_debug_server.py')