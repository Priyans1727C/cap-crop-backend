from celery import shared_task
from celery.utils.log import get_task_logger
from django.conf import settings
from apps.users.services.email_service import EmailService
from apps.users.services.verification_service import VerificationService
from django.contrib.auth import get_user_model



User = get_user_model()
logger = get_task_logger(__name__)

@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def send_verification_email(self,user_id,base_url):
    try:
        user = User.objects.only("id", "email", "is_active", "password","last_login").get(id=user_id)
        VerificationService.send_verification_email(user=user,base_url=base_url)
    except User.DoesNotExist:
         return f"User with id={user_id} does not exist"

    except Exception as exec:
        raise self.retry(exec=exec)

@shared_task(bind=True,max_retries=3,default_retry_delay=60)
def send_verification_password_email(self,user_id,base_url):
    try:
        user =User.objects.only("id","password","email","last_login","is_active").get(id=user_id)
        VerificationService.send_verification_password_email(user=user,base_url=base_url)
    except User.DoesNotExist:
        return f"user with{user_id} doesn't exist"
    except Exception as exce:
        raise self.retry(exce=exce)
    

@shared_task
def test_task():
    print("✅ Test task is working!")
    return "Task completed successfully"


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def send_brevo_test_email(self, to_email=None):
    recipient = to_email or settings.DEFAULT_FROM_EMAIL or settings.BREVO_SENDER_EMAIL
    try:
        logger.info("Sending Brevo test email to %s", recipient)
        response = EmailService.send_email(
            subject="Brevo test email from auth_service",
            to_email=recipient,
            text_content=(
                "This is a test email sent from auth_service to confirm Brevo is working correctly."
            ),
            html_content=(
                "<html><body>"
                "<h2>Brevo test email</h2>"
                "<p>This is a test email sent from auth_service to confirm Brevo is working correctly.</p>"
                "</body></html>"
            ),
        )
        logger.info(
            "Brevo test email sent successfully to %s with response %s",
            recipient,
            response,
        )
        return {
            "detail": "Brevo test email sent successfully.",
            "recipient": recipient,
            "brevo_response": response,
        }
    except Exception as exc:
        logger.exception("Brevo test email failed for %s", recipient)
        raise self.retry(exc=exc)