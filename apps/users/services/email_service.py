from django.conf import settings
import json
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from apps.users.utils.email_templates import welcome_email_template

class EmailService:
    @staticmethod
    def send_email(subject,to_email,text_content,html_content=None):
        api_key = getattr(settings, "BREVO_API_KEY", "")
        if not api_key:
            raise ValueError("BREVO_API_KEY is not configured")

        payload = {
            "sender": {
                "email": settings.BREVO_SENDER_EMAIL,
                "name": settings.BREVO_SENDER_NAME,
            },
            "to": [{"email": to_email}],
            "subject": subject,
            "textContent": text_content,
        }

        if html_content:
            payload["htmlContent"] = html_content

        req = urlrequest.Request(
            url="https://api.brevo.com/v3/smtp/email",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "api-key": api_key,
            },
            method="POST",
        )

        try:
            with urlrequest.urlopen(req, timeout=15) as response:
                response_body = response.read().decode("utf-8", errors="ignore")
                if not response_body:
                    return {"status": response.status, "message": "Email accepted by Brevo"}

                try:
                    return json.loads(response_body)
                except json.JSONDecodeError:
                    return {"status": response.status, "message": response_body}
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Brevo API HTTPError {exc.code}: {error_body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Brevo API URLError: {exc.reason}") from exc
        
    def send_welcome_email(user):
        text_content,html_content =  welcome_email_template(user)
        EmailService.send_email(
            subject="Welcom Message",
            to_email=user.email,
            text_content=text_content,
            html_content=html_content
        )