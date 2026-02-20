#!/usr/bin/env bash
set -o errexit

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Collecting static files..."
python manage.py collectstatic --no-input

echo "Running migrations..."
python manage.py migrate --no-input --run-syncdb

echo "Creating superuser if not exists..."
python manage.py shell << END
import os
from django.contrib.auth import get_user_model

User = get_user_model()

email = os.environ.get("DJANGO_SUPERUSER_EMAIL")
password = os.environ.get("DJANGO_SUPERUSER_PASSWORD")

if email and password:
    if not User.objects.filter(email=email).exists():
        User.objects.create_superuser(email=email, password=password)
        print("✅ Superuser created successfully.")
    else:
        print("⚡ Superuser already exists.")
else:
    print("❌ Superuser environment variables not set.")
END

echo "Build completed successfully!"