def verification_email_template(user, verify_url):
    text_content = f"""
    Hi {user.email},

    Please click the link below to verify your email:
    {verify_url}

    This link will expire in 24 hours.
    """

    html_content = f"""
    <html>
    <body>
        <h2>Hello {user.email},</h2>
        <p>Please verify your email by clicking the button below:</p>
        <p><a href="{verify_url}" 
              style="padding:10px 20px; background:#007bff; color:white; 
                     text-decoration:none; border-radius:5px;">
           Verify Email
        </a></p>
        <p>This link will expire in 24 hours.</p>
    </body>
    </html>
    """

    return text_content, html_content



def welcome_email_template(user):
    text_content = f"""
    Hi {user.email},

    Welcome to ArabDabha!

    We're excited to have you on board. If you have any questions, feel free to reply to this email.

    Best regards,
    The ArabDabha Team
    """

    html_content = f"""
    <html>
    <body>
        <h2>Welcome, {user.email}!</h2>
        <p>We're excited to have you join <b>ArabDabha</b>!</p>
        <p>If you have any questions, just reply to this email.</p>
        <br>
        <p>Best regards,<br>
        The ArabDabha Team</p>
    </body>
    </html>
    """
    
    return text_content, html_content


def password_reset_email_template(user, reset_url):
    text_content = f"""
    Hi {user.email},

    You requested a password reset.
    Please click the link below to reset your password:
    {reset_url}

    If you did not request this, you can safely ignore this email.
    """

    html_content = f"""
    <html>
    <body>
        <h2>Hello {user.email},</h2>
        <p>You requested to reset your password.</p>
        <p><a href="{reset_url}" 
              style="padding:10px 20px; background:#28a745; color:white; 
                     text-decoration:none; border-radius:5px;">
           Reset Password
        </a></p>
        <p>If you did not request this, you can ignore this email.</p>
    </body>
    </html>
    """

    return text_content, html_content
