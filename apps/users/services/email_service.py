from django.core.mail import EmailMultiAlternatives
from django.conf import settings

from apps.users.utils.email_templates import welcome_email_template

class EmailService:
    @staticmethod
    def send_email(subject,to_email,text_content,html_content=None):
        email = EmailMultiAlternatives(
            subject=subject,
            body = text_content,
            from_email = settings.DEFAULT_FROM_EMAIL,
            to = [to_email]
        )
        
        if html_content:
            email.attach_alternative(html_content,"text/html")
        email.send()
        
    def send_welcome_email(user):
        text_content,html_content =  welcome_email_template(user)
        EmailService.send_email(
            subject="Welcom Message",
            to_email=user.email,
            text_content=text_content,
            html_content=html_content
        )