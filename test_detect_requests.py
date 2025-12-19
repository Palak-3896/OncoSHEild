import requests
import time

url = 'http://127.0.0.1:5000'
# Wait for server up
for i in range(10):
    try:
        r = requests.get(url + '/detect', timeout=3)
        print('GET /detect status', r.status_code)
        break
    except Exception as e:
        print('waiting for server...', e)
        time.sleep(1)

r = requests.post(url + '/detect_cancer', data={'radius_mean': 20, 'texture_mean': 25, 'perimeter_mean': 100, 'area_mean':900, 'smoothness_mean':0.2})
print('POST /detect_cancer status', r.status_code)
print('Has Book Appointment button?', 'book-appointment-btn' in r.text)
print('Length of response', len(r.text))

# Test booking endpoint
payload={'name':'Test User','email':'test@example.com','phone':'1234567890','preferred_date':'2025-12-25','detection_text':'⚠ Malignant (Potentially Cancerous)','test_data':{'radius_mean':20},'ai_analysis':{'interpretation':'Test'}}
resp = requests.post(url + '/book_appointment', json=payload)
print('BOOK status', resp.status_code)
try:
    print('BOOK json:', resp.json())
except Exception as e:
    print('BOOK no json', e)
