import requests
url='http://127.0.0.1:5000'
payload={'name':'Test User','email':'test@example.com','phone':'1234567890','preferred_date':'2025-12-25','detection_text':'⚠ Malignant (Potentially Cancerous)','test_data':{'radius_mean':20},'ai_analysis':{'interpretation':'Test'}}
resp = requests.post(url+'/book_appointment', json=payload)
print('status', resp.status_code)
try:
    print(resp.json())
except Exception as e:
    print('no json', e)
