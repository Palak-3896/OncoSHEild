import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class HospitalService:
    """Simple wrapper to call an external hospital booking API using HOSPITAL_API_KEY.
    Falls back to saving locally (reports/appointments.json) and sending confirmation email via SMTP
    if the external API is not configured or fails.
    """

    def __init__(self):
        self.api_key = os.getenv('HOSPITAL_API_KEY')
        self.base_url = os.getenv('HOSPITAL_BASE_URL')  # e.g. https://hospital.example.com/api

        # SMTP fallbacks (optional)
        self.smtp_server = os.getenv('EMAIL_SMTP_SERVER')
        self.smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '587')) if os.getenv('EMAIL_SMTP_PORT') else None
        self.email_user = os.getenv('EMAIL_USER')
        self.email_pass = os.getenv('EMAIL_PASS')

    def book_appointment(self, booking_payload: dict) -> dict:
        """Attempt to book via hospital API, otherwise fallback to local/save + optional SMTP email.
        Returns dict with at least keys: success (bool), booking_id (str|None), message (str).
        """
        # Try hospital API if configured
        email_sent = False
        email_message = None

        if self.api_key and self.base_url:
            try:
                import requests
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
                url = self.base_url.rstrip('/') + '/appointments'
                resp = requests.post(url, json=booking_payload, headers=headers, timeout=10)
                if resp.status_code in (200, 201):
                    try:
                        data = resp.json()
                    except Exception:
                        data = {}
                    booking_id = data.get('id') or data.get('booking_id') or str(uuid.uuid4())

                    # Always try to send a confirmation email from our side if SMTP is configured and an email is provided
                    try:
                        to_email = booking_payload.get('email')
                        if to_email and self.smtp_server and self.email_user and self.email_pass and self.smtp_port:
                            email_sent = self._send_confirmation_email(to_email, booking_id, booking_payload)
                            email_message = 'Confirmation email sent' if email_sent else 'Confirmation email failed to send'
                    except Exception as e:
                        print('Error trying to send confirmation email after API success:', e)

                    return {'success': True, 'booking_id': booking_id, 'message': data.get('message', 'Booked via hospital API'), 'email_sent': email_sent, 'email_message': email_message}
                else:
                    return {'success': False, 'message': f'Hospital API error {resp.status_code}: {resp.text}'}
            except Exception as e:
                # continue to fallback
                fallback_reason = f'Hospital API request failed: {e}'
                print(fallback_reason)
                return self._fallback_save(booking_payload, fallback_reason)

        # Fallback if API not configured
        return self._fallback_save(booking_payload, 'Hospital API not configured')

    def _fallback_save(self, payload, reason):
        # Ensure reports folder exists
        os.makedirs('reports', exist_ok=True)
        appt_file = os.path.join('reports', 'appointments.json')

        try:
            existing = []
            if os.path.exists(appt_file):
                with open(appt_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
        except Exception:
            existing = []

        booking_id = str(uuid.uuid4())
        record = {
            'booking_id': booking_id,
            'payload': payload,
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'note': reason
        }
        existing.append(record)
        with open(appt_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        # Try to send confirmation email (if SMTP configured)
        email_sent = False
        email_message = None
        try:
            to_email = payload.get('email')
            if to_email and self.smtp_server and (self.email_user or self.smtp_port):
                # send even if there is no auth (some local SMTPs accept unauthenticated sends)
                email_sent = self._send_confirmation_email(to_email, booking_id, payload)
                email_message = 'Confirmation email sent' if email_sent else 'Failed to send confirmation email'
        except Exception as e:
            print('Email fallback failure:', e)

        message = f'Appointment saved locally (id={booking_id}). '
        message += email_message if email_message else 'No confirmation email sent (SMTP not configured or failed).'
        return {'success': True, 'booking_id': booking_id, 'message': message, 'email_sent': email_sent, 'email_message': email_message}

    def _send_confirmation_email(self, to_email, booking_id, payload):
        """Send a confirmation email to `to_email`. Returns True on success, False otherwise."""
        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            # Build nice email content (plain + HTML)
            hospital_name = os.getenv('HOSPITAL_DISPLAY_NAME') or os.getenv('HOSPITAL_CONTACT_NAME') or 'Your Hospital'
            subject = f'Appointment Request Received - {hospital_name} - {booking_id}'

            plain_body = (
                f"Dear {payload.get('name','Patient')},\n\n"
                f"We have received your appointment request.\n\n"
                f"Booking ID: {booking_id}\n\n"
                f"Requested date/time: {payload.get('preferred_date','Not specified')}\n\n"
                f"Detection result: {payload.get('detection_text','N/A')}\n\n"
                f"We will contact you shortly to confirm the final appointment details.\n\n"
                f"If you have questions, please reply to this email.\n\n"
                f"Regards,\n{hospital_name}"
            )

            html_body = f"""
            <html>
            <body>
              <p>Dear {payload.get('name','Patient')},</p>
              <p>We have received your appointment request.</p>
              <p><strong>Booking ID:</strong> {booking_id}<br>
                 <strong>Requested date/time:</strong> {payload.get('preferred_date','Not specified')}<br>
                 <strong>Detection result:</strong> {payload.get('detection_text','N/A')}</p>
              <p>We will contact you shortly to confirm the final appointment details.</p>
              <p>Regards,<br>{hospital_name}</p>
            </body>
            </html>
            """

            # Sender email
            sender = os.getenv('EMAIL_FROM') or self.email_user or os.getenv('HOSPITAL_CONTACT_EMAIL') or 'no-reply@localhost'

            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = sender
            msg['To'] = to_email

            part1 = MIMEText(plain_body, 'plain')
            part2 = MIMEText(html_body, 'html')
            msg.attach(part1)
            msg.attach(part2)

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.ehlo()
            # Try TLS if port suggests it
            try:
                server.starttls()
                server.ehlo()
            except Exception:
                # TLS may not be supported; continue
                pass

            # Login if credentials are available
            if self.email_user and self.email_pass:
                try:
                    server.login(self.email_user, self.email_pass)
                except Exception as e:
                    print('SMTP login failed:', e)
                    # proceed to try sending anyway

            try:
                server.sendmail(sender, [to_email], msg.as_string())
                server.quit()
                return True
            except Exception as e:
                print('SMTP send failed:', e)
                try:
                    server.quit()
                except Exception:
                    pass
                return False
        except Exception as e:
            print('General email error:', e)
            return False
