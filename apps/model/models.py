from django.db import models


class AgricultureFeedCache(models.Model):
	class FeedType(models.TextChoices):
		NEWS = "news", "News"
		SCHEMES = "schemes", "Schemes"

	feed_type = models.CharField(max_length=20, choices=FeedType, unique=True)
	payload = models.JSONField(default=list)
	expires_at = models.DateTimeField(db_index=True)
	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	class Meta:
		db_table = "agriculture_feed_cache"

	def __str__(self):
		return f"{self.feed_type} cache until {self.expires_at.isoformat()}"
