# In your core/celery.py

import sys

# Only apply the monkey patch when the process is a Celery worker.
# We can detect this by checking the command-line arguments.
is_celery_worker = sys.argv and 'celery' in sys.argv[0]

if is_celery_worker:
    import eventlet
    eventlet.monkey_patch()

import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

app = Celery('core')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()