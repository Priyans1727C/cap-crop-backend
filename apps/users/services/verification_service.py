from .token_service import EmailVerificationToken
from .email_service import EmailService

from django.urls import reverse
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes

from apps.users.utils.email_templates import verification_email_template, password_reset_email_template


class VerificationService:
    
    @staticmethod
    def send_verification_email(user,base_url=None,request=None):
        uid = urlsafe_base64_encode(force_bytes(user.pk))
        token = EmailVerificationToken.create(user)
        
        if base_url:
            verify_url = f"{base_url.rstrip('/')}{reverse('verify_email', kwargs={'uidb64': uid, 'token': token})}"
            
        if request:
            verify_url = request.build_absolute_uri(
                reverse("verify_email",kwargs={"uidb64":uid,"token":token})
            )
        
        
        text_content,html_content = verification_email_template(user,verify_url)
        EmailService.send_email(
            subject="Verify your Email",
            to_email=user.email,
            text_content=text_content,
            html_content=html_content
        )
        
        
    @staticmethod
    def send_verification_password_email(user,request=None,base_url=None):
        uid = urlsafe_base64_encode(force_bytes(user.pk))
        token = EmailVerificationToken.create(user)
        if request:
            reset_url = request.build_absolute_uri(
                reverse("password_reset_confirm",kwargs={"uidb64":uid,"token":token})
            )
        if base_url:
            reset_url=  f"{base_url.rstrip('/')}{reverse('password_reset_confirm', kwargs={'uidb64': uid, 'token': token})}"
            
        text_content, html_content = password_reset_email_template(user, reset_url)
        EmailService.send_email(
            subject="Reset your Password",
            to_email=user.email,
            text_content=text_content,
            html_content=html_content
        )
        
    @staticmethod
    def verify_token(user,token):
        return EmailVerificationToken.verify(user,token)
        
    @staticmethod
    def activate_user(user,token):
        if EmailVerificationToken.verify(user,token):
            # user.is_active=True
            user.is_email_verified=True
            user.save()
            return True
        return False