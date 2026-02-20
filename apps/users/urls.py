from django.urls import path
from .views import UserView,UserRegisterView,UserLoginView,UserChangePasswordView,AccessTokenRefreshView,UserLogoutView,VerifyEmailView,ForgotPasswordView,ResetPasswordView
from .views import UserProfileViewSet


urlpatterns = [
    path("me/",UserProfileViewSet.as_view(),name="profile"),
    path("refresh/", AccessTokenRefreshView.as_view(), name="refresh"),
    path("register/", UserRegisterView.as_view(), name="register"),
    path('login/', UserLoginView.as_view()),
    path("change-password/", UserChangePasswordView.as_view(), name="change_password"),
    path("logout/", UserLogoutView.as_view(), name="logout"),
    path("verify_email/<str:uidb64>/<str:token>/",VerifyEmailView.as_view(),name="verify_email"),
    path("forgot-password/", ForgotPasswordView.as_view(), name="forgot_password"),
    path("reset-password/<uidb64>/<token>/", ResetPasswordView.as_view(), name="password_reset_confirm"),
    
   path("healthz/", lambda r: __import__("django.http").http.JsonResponse({"ok": True})),
]
# urlpatterns+=router.urls