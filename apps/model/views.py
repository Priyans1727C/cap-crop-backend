
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, serializers
from .ml.ml_model import CropRecommender

class CropInputSerializer(serializers.Serializer):
	N = serializers.FloatField()
	P = serializers.FloatField()
	K = serializers.FloatField()
	temperature = serializers.FloatField()
	humidity = serializers.FloatField()
	ph = serializers.FloatField()
	rainfall = serializers.FloatField()

class CropRecommendationAPIView(APIView):
	recommender = None

	def get_recommender(self):
		if not self.recommender:
			self.recommender = CropRecommender()
		return self.recommender

	def post(self, request):
		serializer = CropInputSerializer(data=request.data)
		if serializer.is_valid():
			data = serializer.validated_data
			user_input = [
				data['N'],
				data['P'],
				data['K'],
				data['temperature'],
				data['humidity'],
				data['ph'],
				data['rainfall']
			]
			recommender = self.get_recommender()
			crop, confidence, alternatives = recommender.recommend_crop_with_alternatives(user_input)
			return Response({
				'recommended_crop': crop,
				'confidence': confidence,
				'alternatives': [
					{'crop': alt[0], 'confidence': alt[1]} for alt in alternatives
				]
			})
		return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
