from celery import shared_task
import time
from apps.users.services.verification_service import VerificationService
from django.contrib.auth import get_user_model



User = get_user_model()

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