from django.urls import path
from .views import CropRecommendationAPIView

urlpatterns = [
	path('rc/', CropRecommendationAPIView.as_view(), name='recommend-crop'),
]
