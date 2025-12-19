import requests

url='http://127.0.0.1:5000'
payload={'name':'SMTP Test','email':'recipient@example.com','phone':'000','preferred_date':'2025-12-30','detection_text':'⚠ Malignant (Potentially Cancerous)','test_data':{'radius_mean':20},'ai_analysis':{'interpretation':'Test'}}
resp = requests.post(url+'/book_appointment', json=payload)
print('status', resp.status_code)
print(resp.json())

# Note for local testing: run a local debug SMTP server in a separate terminal:
# python -m smtpd -c DebuggingServer -n localhost:1025
# Then set in .env:
# EMAIL_SMTP_SERVER=localhost
# EMAIL_SMTP_PORT=1025
# EMAIL_FROM=no-reply@local.test
# (no auth required for the debug server)
