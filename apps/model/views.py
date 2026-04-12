import json
import logging
import re
from datetime import timedelta
from html import unescape
from threading import Lock
from urllib import error as urllib_error
from urllib import request as urllib_request
from xml.etree import ElementTree as ET
from decouple import config
from django.utils import timezone

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, serializers
from .models import AgricultureFeedCache


logger = logging.getLogger(__name__)

class CropInputSerializer(serializers.Serializer):
	N = serializers.FloatField()
	P = serializers.FloatField()
	K = serializers.FloatField()
	temperature = serializers.FloatField()
	humidity = serializers.FloatField()
	ph = serializers.FloatField()
	rainfall = serializers.FloatField()


class GeminiChatSerializer(serializers.Serializer):
	message = serializers.CharField(max_length=4000, allow_blank=False, trim_whitespace=True)

class CropRecommendationAPIView(APIView):
	recommender = None
	recommender_lock = Lock()

	def get_recommender(self):
		cls = type(self)
		if cls.recommender is None:
			with cls.recommender_lock:
				if cls.recommender is None:
					# Import lazily so URL resolver/health checks do not trigger heavy ML imports.
					from .ml.ml_model import CropRecommender
					cls.recommender = CropRecommender()
		return cls.recommender

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
			try:
				recommender = self.get_recommender()
			except Exception:
				logger.exception("Failed to initialize crop recommender")
				return Response(
					{"error": "Crop recommendation service is temporarily unavailable."},
					status=status.HTTP_503_SERVICE_UNAVAILABLE,
				)
			crop, confidence, alternatives = recommender.recommend_crop_with_alternatives(user_input)
			return Response({
				'recommended_crop': crop,
				'confidence': confidence,
				'alternatives': [
					{'crop': alt[0], 'confidence': alt[1]} for alt in alternatives
				]
			})
		return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class GeminiAgricultureFeedService:
	CACHE_TTL_HOURS = 6
	RSS_TIMEOUT_SECONDS = 20

	def _get_api_key(self):
		return config("GEMINI_API_KEY", default="").strip()

	def _get_model(self):
		return config("GEMINI_MODEL", default="gemini-3-flash-preview").strip()

	def _cache_get(self, feed_type, include_expired=False):
		now = timezone.now()
		entry = AgricultureFeedCache.objects.filter(feed_type=feed_type).first()
		if not entry:
			return None
		if entry.expires_at <= now and not include_expired:
			return None
		return entry.payload

	def _cache_set(self, feed_type, payload):
		expires_at = timezone.now() + timedelta(hours=self.CACHE_TTL_HOURS)
		AgricultureFeedCache.objects.update_or_create(
			feed_type=feed_type,
			defaults={"payload": payload, "expires_at": expires_at},
		)

	def _cleanup_expired(self):
		"""Delete expired cache entries from database."""
		now = timezone.now()
		deleted_count, _ = AgricultureFeedCache.objects.filter(expires_at__lt=now).delete()
		return deleted_count

	def _extract_text(self, response_payload):
		candidates = response_payload.get("candidates") or []
		if not candidates:
			raise ValueError("No candidates returned from Gemini")

		parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
		if not parts:
			raise ValueError("No content parts returned from Gemini")

		return str(parts[0].get("text") or "")

	def _call_gemini(self, prompt):
		api_key = self._get_api_key()
		if not api_key:
			raise ValueError("GEMINI_API_KEY is not configured")

		model = self._get_model()
		url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

		payload = {
			"contents": [
				{
					"parts": [
						{"text": prompt}
					]
				}
			],
			"generationConfig": {
				"responseMimeType": "application/json",
			}
		}

		request_body = json.dumps(payload).encode("utf-8")
		req = urllib_request.Request(
			url=url,
			data=request_body,
			headers={"Content-Type": "application/json"},
			method="POST",
		)

		try:
			with urllib_request.urlopen(req, timeout=60) as resp:
				response_data = json.loads(resp.read().decode("utf-8"))
		except urllib_error.HTTPError as exc:
			message = exc.read().decode("utf-8", errors="ignore")
			raise ValueError(f"Gemini HTTP error: {message}") from exc
		except urllib_error.URLError as exc:
			raise ValueError("Could not reach Gemini API") from exc

		text = self._extract_text(response_data).strip()
		if text.startswith("```"):
			text = text.strip("`")
			if text.startswith("json"):
				text = text[4:].strip()

		try:
			return json.loads(text)
		except json.JSONDecodeError as exc:
			raise ValueError("Gemini returned non-JSON content") from exc

	def _call_gemini_text(self, prompt):
		api_key = self._get_api_key()
		if not api_key:
			raise ValueError("GEMINI_API_KEY is not configured")

		model = self._get_model()
		url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

		payload = {
			"contents": [
				{
					"parts": [
						{"text": prompt}
					]
				}
			]
		}

		request_body = json.dumps(payload).encode("utf-8")
		req = urllib_request.Request(
			url=url,
			data=request_body,
			headers={"Content-Type": "application/json"},
			method="POST",
		)

		try:
			with urllib_request.urlopen(req, timeout=60) as resp:
				response_data = json.loads(resp.read().decode("utf-8"))
		except urllib_error.HTTPError as exc:
			message = exc.read().decode("utf-8", errors="ignore")
			raise ValueError(f"Gemini HTTP error: {message}") from exc
		except urllib_error.URLError as exc:
			raise ValueError("Could not reach Gemini API") from exc

		text = self._extract_text(response_data).strip()
		if not text:
			raise ValueError("Gemini returned empty response")
		return text

	def _strip_html(self, text):
		if not text:
			return ""
		text = re.sub(r"<[^>]+>", " ", str(text))
		return re.sub(r"\s+", " ", unescape(text)).strip()

	def _infer_scheme_type(self, title, summary):
		content = f"{title} {summary}".lower()
		if "state" in content:
			return "State"
		return "Central"

	def _infer_scheme_category(self, title, summary):
		content = f"{title} {summary}".lower()
		if "insurance" in content:
			return "Insurance"
		return "Subsidy"

	def _extract_benefits(self, summary):
		text = self._strip_html(summary)
		if not text:
			return []
		parts = [p.strip() for p in re.split(r"[.;]\s+", text) if p.strip()]
		return parts[:3]

	def _infer_news_category(self, title, summary):
		content = f"{title} {summary}".lower()
		if any(x in content for x in ["policy", "ministry", "government", "cabinet"]):
			return "Policy"
		if any(x in content for x in ["market", "price", "mandi", "export", "import"]):
			return "Market"
		if any(x in content for x in ["rain", "drought", "climate", "weather"]):
			return "Climate"
		if any(x in content for x in ["tech", "drone", "ai", "digital", "innovation"]):
			return "Technology"
		return "Research"

	def _fetch_news_from_rss(self):
		"""Fetch real-time agriculture news from public RSS search feeds."""
		rss_urls = [
			"https://news.google.com/rss/search?q=India+agriculture+news&hl=en-IN&gl=IN&ceid=IN:en",
			"https://news.google.com/rss/search?q=Indian+farming+policy+market+update&hl=en-IN&gl=IN&ceid=IN:en",
		]

		items = []
		seen_links = set()

		for rss_url in rss_urls:
			try:
				with urllib_request.urlopen(rss_url, timeout=self.RSS_TIMEOUT_SECONDS) as resp:
					raw_xml = resp.read()
			except Exception:
				continue

			try:
				root = ET.fromstring(raw_xml)
			except ET.ParseError:
				continue

			for entry in root.findall("./channel/item"):
				title = (entry.findtext("title") or "").strip()
				link = (entry.findtext("link") or "").strip()
				description = (entry.findtext("description") or "").strip()
				pub_date = (entry.findtext("pubDate") or "").strip()

				if not title or not link or link in seen_links:
					continue

				seen_links.add(link)
				summary = self._strip_html(description)[:320]

				items.append(
					{
						"id": len(items) + 1,
						"title": title,
						"summary": summary,
						"category": self._infer_news_category(title, summary),
						"date": pub_date[:16],
						"readTime": "4 min",
						"tag": "new",
						"emoji": "🌾",
						"url": link,
					}
				)

				if len(items) >= 8:
					break

			if len(items) >= 8:
				break

		return items[:8]

	def _fetch_schemes_from_rss(self):
		"""Fetch real-time agriculture scheme updates from public RSS search feeds."""
		rss_urls = [
			"https://news.google.com/rss/search?q=Indian+agriculture+government+schemes&hl=en-IN&gl=IN&ceid=IN:en",
			"https://news.google.com/rss/search?q=PM-KISAN+or+crop+insurance+scheme+India&hl=en-IN&gl=IN&ceid=IN:en",
		]

		items = []
		seen_links = set()

		for rss_url in rss_urls:
			try:
				with urllib_request.urlopen(rss_url, timeout=self.RSS_TIMEOUT_SECONDS) as resp:
					raw_xml = resp.read()
			except Exception:
				continue

			try:
				root = ET.fromstring(raw_xml)
			except ET.ParseError:
				continue

			for entry in root.findall("./channel/item"):
				title = (entry.findtext("title") or "").strip()
				link = (entry.findtext("link") or "").strip()
				description = (entry.findtext("description") or "").strip()
				pub_date = (entry.findtext("pubDate") or "").strip()

				if not title or not link or link in seen_links:
					continue

				seen_links.add(link)
				summary = self._strip_html(description)[:350]
				benefits = self._extract_benefits(description)

				items.append(
					{
						"id": len(items) + 1,
						"name": title[:80],
						"fullName": title,
						"status": "Ongoing",
						"type": self._infer_scheme_type(title, summary),
						"category": self._infer_scheme_category(title, summary),
						"amount": "",
						"deadline": None,
						"launch": pub_date[:16],
						"description": summary,
						"eligibility": "Refer official notification for eligibility criteria.",
						"benefits": benefits,
						"icon": "🌾",
						"link": link,
					}
				)

				if len(items) >= 8:
					break

			if len(items) >= 8:
				break

		return items[:8]

	def get_chat_reply(self, message):
		prompt = (
			"You are a concise agriculture assistant for Indian farmers. "
			"Give practical, safe and clear guidance.\n\n"
			f"User question: {message}"
		)
		return self._call_gemini_text(prompt)

	def get_latest_news(self):
		cache_key = AgricultureFeedCache.FeedType.NEWS

		# Clean up expired records first
		self._cleanup_expired()

		# Check for valid (non-expired) cache
		cached = self._cache_get(cache_key)
		if cached is not None:
			return cached

		# Try to get stale data as fallback
		stale_payload = self._cache_get(cache_key, include_expired=True)

		prompt = (
			"Return ONLY JSON with this exact structure: "
			"{\"items\":[{\"id\":1,\"title\":\"\",\"summary\":\"\",\"category\":\"Technology|Policy|Market|Research|Climate|Export\",\"date\":\"Apr 05, 2026\",\"readTime\":\"4 min\",\"tag\":\"trending|new|hot|important|null\",\"emoji\":\"🌾\",\"url\":\"https://...\"}]}. "
			"Give 8 latest agriculture news items focused on India. Use concise factual summaries."
		)

		try:
			response_payload = self._call_gemini(prompt)
			items = response_payload.get("items")
			if not isinstance(items, list):
				raise ValueError("Invalid Gemini news response format")

			cleaned_items = []
			for index, item in enumerate(items, start=1):
				if not isinstance(item, dict):
					continue
				cleaned_items.append(
					{
						"id": int(item.get("id") or index),
						"title": str(item.get("title") or ""),
						"summary": str(item.get("summary") or ""),
						"category": str(item.get("category") or "Research"),
						"date": str(item.get("date") or ""),
						"readTime": str(item.get("readTime") or "4 min"),
						"tag": item.get("tag"),
						"emoji": str(item.get("emoji") or "🌱"),
						"url": str(item.get("url") or ""),
					}
				)

			self._cache_set(cache_key, cleaned_items)
			return cleaned_items
		except Exception as exc:
			if stale_payload is not None:
				return stale_payload

			fallback_items = self._fetch_news_from_rss()
			if fallback_items:
				self._cache_set(cache_key, fallback_items)
				return fallback_items

			raise ValueError(
				f"Could not fetch news from Gemini API and no cached data available: {exc}"
			)

	def get_latest_schemes(self):
		cache_key = AgricultureFeedCache.FeedType.SCHEMES

		# Clean up expired records first
		self._cleanup_expired()

		# Check for valid (non-expired) cache
		cached = self._cache_get(cache_key)
		if cached is not None:
			return cached

		# Try to get stale data as fallback
		stale_payload = self._cache_get(cache_key, include_expired=True)

		prompt = (
			"Return ONLY JSON with this exact structure: "
			"{\"items\":[{\"id\":1,\"name\":\"\",\"fullName\":\"\",\"status\":\"Ongoing|Upcoming\",\"type\":\"Central|State\",\"category\":\"Subsidy|Insurance\",\"amount\":\"\",\"deadline\":\"Mar 31, 2026|null\",\"launch\":\"Feb 2025\",\"description\":\"\",\"eligibility\":\"\",\"benefits\":[\"\",\"\",\"\"],\"icon\":\"🌱\",\"link\":\"https://...\"}]}. "
			"Give 8 latest real Indian agriculture government schemes and updates."
		)

		try:
			response_payload = self._call_gemini(prompt)
			items = response_payload.get("items")
			if not isinstance(items, list):
				raise ValueError("Invalid Gemini schemes response format")

			cleaned_items = []
			for index, item in enumerate(items, start=1):
				if not isinstance(item, dict):
					continue
				benefits = item.get("benefits")
				if not isinstance(benefits, list):
					benefits = []

				cleaned_items.append(
					{
						"id": int(item.get("id") or index),
						"name": str(item.get("name") or ""),
						"fullName": str(item.get("fullName") or ""),
						"status": str(item.get("status") or "Ongoing"),
						"type": str(item.get("type") or "Central"),
						"category": str(item.get("category") or "Subsidy"),
						"amount": str(item.get("amount") or ""),
						"deadline": item.get("deadline"),
						"launch": str(item.get("launch") or ""),
						"description": str(item.get("description") or ""),
						"eligibility": str(item.get("eligibility") or ""),
						"benefits": [str(x) for x in benefits][:5],
						"icon": str(item.get("icon") or "🌱"),
						"link": str(item.get("link") or ""),
					}
				)

			self._cache_set(cache_key, cleaned_items)
			return cleaned_items
		except Exception as exc:
			if stale_payload is not None:
				return stale_payload

			fallback_items = self._fetch_schemes_from_rss()
			if fallback_items:
				self._cache_set(cache_key, fallback_items)
				return fallback_items

			raise ValueError(
				f"Could not fetch schemes from Gemini API and no cached data available: {exc}"
			)


class AgricultureNewsAPIView(APIView):
	service = GeminiAgricultureFeedService()

	def get(self, request):
		try:
			items = self.service.get_latest_news()
			return Response({"items": items}, status=status.HTTP_200_OK)
		except Exception as exc:
			return Response(
				{"detail": "Could not fetch agriculture news from Gemini", "error": str(exc)},
				status=status.HTTP_503_SERVICE_UNAVAILABLE,
			)


class AgricultureSchemesAPIView(APIView):
	service = GeminiAgricultureFeedService()

	def get(self, request):
		try:
			items = self.service.get_latest_schemes()
			return Response({"items": items}, status=status.HTTP_200_OK)
		except Exception as exc:
			return Response(
				{"detail": "Could not fetch agriculture schemes from Gemini", "error": str(exc)},
				status=status.HTTP_503_SERVICE_UNAVAILABLE,
			)


class GeminiChatAPIView(APIView):
	service = GeminiAgricultureFeedService()

	def post(self, request):
		serializer = GeminiChatSerializer(data=request.data)
		if not serializer.is_valid():
			return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

		message = serializer.validated_data["message"]
		try:
			reply = self.service.get_chat_reply(message)
			return Response({"reply": reply}, status=status.HTTP_200_OK)
		except Exception as exc:
			return Response(
				{"detail": "Could not fetch chat response from Gemini", "error": str(exc)},
				status=status.HTTP_503_SERVICE_UNAVAILABLE,
			)
