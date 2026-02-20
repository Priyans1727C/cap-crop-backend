from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.contrib.auth import get_user_model
from rest_framework import status
from .serializers import UserSafeSerializer,UpdateUserSerializer,UserRegisterSerializer,RoleTokenObtainPairSerializer,UserChangePassword,UserEmailSerializer,ResetPasswordSerializer,UserProfileSerializer
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.generics import RetrieveUpdateAPIView
from rest_framework.permissions import IsAuthenticated


from django.contrib.auth import authenticate,login
from rest_framework.decorators import action
from rest_framework.authtoken.models import Token 
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .auth import set_refresh_cookie,clear_refresh_cookie
from rest_framework_simplejwt.tokens import RefreshToken

from apps.users.services.email_service import EmailService
from apps.users.services.verification_service import VerificationService
from django.utils.http import urlsafe_base64_decode
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from .throttling import (LoginBurstThrottle, LoginSustainedThrottle)
# Create your views here.
from .tasks import send_verification_email,send_verification_password_email

User = get_user_model()
from .models import UserProfile


class UserView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self,request):
        ser = UserSafeSerializer(request.user)
        return Response(ser.data)
    
    def patch(self,request):
        ser = UpdateUserSerializer(request.user, data=request.data, partial=True)
        ser.is_valid(raise_exception=True)
        ser.save()
        return Response(ser.data)
    
class UserProfileViewSet(RetrieveUpdateAPIView):
    throttle_scope = "user_profile"
    serializer_class = UserProfileSerializer
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def get_object(self):
        profile, _ = UserProfile.objects.get_or_create(user=self.request.user)
        return profile
    
class UserRegisterView(APIView):
    throttle_scope = "register"
    def post(self,request):
        ser = UserRegisterSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        user = ser.save()
        base_url = request.build_absolute_uri("/")
        send_verification_email.delay(user_id=user.id, base_url=base_url)
        return Response(
            {
                "detail":"User registered successfully. Please verify your email.",
                "user":UserSafeSerializer(user).data,
            },
            status=status.HTTP_201_CREATED
        )
        
    

class VerifyEmailView(APIView):
    throttle_scope = "verify_email"
    def get(self,request,uidb64,token):
        try:
            uid = urlsafe_base64_decode(uidb64)
            user = User.objects.get(pk=uid)
        except Exception:
            return Response({"detail":"Invalid or expired link"},status=status.HTTP_400_BAD_REQUEST)

        if VerificationService.activate_user(user,token):
             return Response({"Email verified successfully!"})
        
        return Response({"detail":"Invalid or expired link"},status=status.HTTP_400_BAD_REQUEST)
"""
#basic session base login

class UserLoginView(APIView):
    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')
        
        user = authenticate(request, email=email, password=password)
        
        if user is not None:
            login(request,user)
            return Response({'token': "nicelydone"})
        else:
            return Response({'error': 'Invalid Credentials'}, status=status.HTTP_400_BAD_REQUEST)
            
"""

class ForgotPasswordView(APIView):
    throttle_scope = "forgot_password"
    serializer_class=UserEmailSerializer 
    def post(self,request,*args, **kwargs):
        ser = self.serializer_class(data=request.data)
        ser.is_valid(raise_exception=True)
        base_url = request.build_absolute_uri("/")
        try:
            user = User.objects.get(email=ser.validated_data['email'])
            print(user)
            send_verification_password_email.delay(user_id=user.id,base_url=base_url)
        except Exception as e:
            pass
        return Response({"detail": "If this email exists, a reset link has been sent."})

class ResetPasswordView(APIView):
    throttle_scope = "verify_reset_password"
    def post(self,request,uidb64,token, **kwargs):
        ser = ResetPasswordSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        try:
            uid = urlsafe_base64_decode(uidb64)
            user = User.objects.get(pk=uid)
        except Exception:
            return Response({"detail":"Invalid or expired link"}, status=status.HTTP_400_BAD_REQUEST)

        if VerificationService.verify_token(user,token):
            user.set_password(ser.validated_data["new_password"])
            user.save(update_fields = ["password"])
            return Response({"detail":"Password changed"},status=200)

        return Response({"detail":"Invalid or expired link"}, status=status.HTTP_400_BAD_REQUEST)
    
        

class AccessTokenRefreshView(TokenRefreshView):
    def post(self, request, *args, **kwargs):
        refresh_token = request.COOKIES.get("refresh_token")
        if not refresh_token:
            return Response({"detail": "No refresh token provided."},status=status.HTTP_400_BAD_REQUEST)
        data = request.data.copy()
        data["refresh"] = refresh_token
        serializer = self.get_serializer(data=data)
        try:
            serializer.is_valid(raise_exception=True)
        except Exception as e:
            return Response({"detail": "Invalid refresh token."}, status=status.HTTP_400_BAD_REQUEST)
        return Response(serializer.validated_data, status=status.HTTP_200_OK)
        # return super().post(request, *args, **kwargs)


class UserLoginView(TokenObtainPairView):
    throttle_scope ="login"
    # throttle_classes = [LoginBurstThrottle,LoginSustainedThrottle]
    serializer_class = RoleTokenObtainPairSerializer
    def post(self, request,*args,**kargs):
        res = super().post(request,*args,**kargs)
        refresh = res.data.pop("refresh",None)
        if refresh:
            set_refresh_cookie(res,refresh)
        return res
        
        
class UserChangePasswordView(APIView):
    throttle_scope = "change_password"
    def post(self,request):
        ser = UserChangePassword(data = request.data)
        ser.is_valid(raise_exception=True)
        if not request.user.check_password(ser.validated_data["old_password"]):
            return Response({"detail":"Old password is incorrect"}, status=400)
        request.user.set_password(ser.validated_data["new_password"])
        request.user.save(update_fields = ["password"])
        return Response({"detail":"Password changed"},status=200)
    
    
class UserLogoutView(APIView):

    permission_classes = [IsAuthenticated]
    def post(self,request):
        rt = request.COOKIES.get("refresh_token")
        if rt:
            try:
                RefreshToken(rt).blacklist()
            except:
                pass
        res = Response({"Detail":"Logout Sucessfull"})
        clear_refresh_cookie(res)
        return res
                