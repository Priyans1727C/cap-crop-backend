from django.urls import path
from .views import CropRecommendationAPIView, AgricultureNewsAPIView, AgricultureSchemesAPIView, GeminiChatAPIView

urlpatterns = [
	path('rc/', CropRecommendationAPIView.as_view(), name='recommend-crop'),
	path('news/', AgricultureNewsAPIView.as_view(), name='agri-news'),
	path('schemes/', AgricultureSchemesAPIView.as_view(), name='agri-schemes'),
	path('chat/', GeminiChatAPIView.as_view(), name='gemini-chat'),
]
