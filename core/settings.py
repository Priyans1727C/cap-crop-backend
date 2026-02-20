from pathlib import Path
from decouple import config, Csv
from datetime import timedelta

BASE_DIR = Path(__file__).resolve().parent.parent

# -------------------------
# SECURITY
# -------------------------
SECRET_KEY = config('SECRET_KEY')
DEBUG = config('DEBUG', default=False, cast=bool)
ALLOWED_HOSTS = config('ALLOWED_HOSTS', default='', cast=Csv())

# -------------------------
# APPLICATIONS
# -------------------------
DJANGO_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]

THIRD_PARTY_APPS = [
    # Add packages like 'rest_framework' here
    # "django.contrib.staticfiles",
    # "daphne",
]

LOCAL_APPS = [
    # Your own apps here
    
    "rest_framework",
    "rest_framework_simplejwt",
    "rest_framework_simplejwt.token_blacklist",
    'django_filters',
    'apps.users',
    'apps.model'

]


INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

# -------------------------
# MIDDLEWARE
# -------------------------
DEFAULT_MIDDLEWARES = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

THIRD_PARTY_MIDDLEWARES=[
    'whitenoise.middleware.WhiteNoiseMiddleware',
]

AUTH_MIDDLEWARES = [
    "corsheaders.middleware.CorsMiddleware",
]

MIDDLEWARE = DEFAULT_MIDDLEWARES+AUTH_MIDDLEWARES+THIRD_PARTY_MIDDLEWARES


ROOT_URLCONF = 'core.urls'

# -------------------------
# TEMPLATES
# -------------------------
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],  # optional: central templates folder
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# WSGI_APPLICATION = 'core.wsgi.application'
ASGI_APPLICATION = 'core.asgi.application'

# -------------------------
# DATABASE
# -------------------------
DATABASES = {
    'default': {
        'ENGINE': config('DB_ENGINE', default='django.db.backends.sqlite3'),
        'NAME': BASE_DIR / config('DB_NAME', default='db.sqlite3'),
    }
}

# -------------------------
# PASSWORDS
# -------------------------
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# -------------------------
# INTERNATIONALIZATION
# -------------------------
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# -------------------------
# STATIC & MEDIA
# -------------------------
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = []

static_dir = BASE_DIR / 'static'
if static_dir.exists():
    STATICFILES_DIRS.append(static_dir)

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# -------------------------
# DEFAULTS
# -------------------------
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# -------------------------
# AUTHENCATION
# -------------------------
AUTH_USER_MODEL = "users.User"

# -------------------------
# REST_FRAMEWORK
# -------------------------
REST_FRAMEWORK = {
    # == Authentication ==
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),

    # == Permissions ==
    #
    #
    
    # == PAGINATION ==
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10,
    
    # == DEFAULT_FILTER ==
    'DEFAULT_FILTER_BACKENDS': ['django_filters.rest_framework.DjangoFilterBackend'],
    
    # == Throttling ==
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',  
        'rest_framework.throttling.ScopedRateThrottle', 
        'apps.users.throttling.BurstRateThrottle',
        'apps.users.throttling.SustainedRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        "anon": "100/min",
        "burst":"200/min",
        "sustained":"500/day",
        "change_password":"10/hour",
        # "register":"6/hour",
        "register":"10/min",
        "verify_email":"2/hour",
        "forgot_password":"3/hour",
        "verify_reset_password":"3/hour",
        "login":"4/minute",
        # "login_sustained":"10/hour",
        "user_profile":"10/minute",
        "resto_details":"8/minute",
    },
}


# -------------------------
# JWT
# -------------------------
SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=config("ACCESS_TOKEN_LIFETIME_MINUTES", cast=int, default=15)),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=config("REFRESH_TOKEN_LIFETIME_DAYS", cast=int, default=7)),
    "ROTATE_REFRESH_TOKENS": config("ROTATE_REFRESH_TOKENS", cast=bool, default=False),
    "BLACKLIST_AFTER_ROTATION": config("BLACKLIST_AFTER_ROTATION", cast=bool, default=True),
    "AUTH_HEADER_TYPES": config("AUTH_HEADER_TYPES", cast=Csv(), default="Bearer"),
    "UPDATE_LAST_LOGIN": config("UPDATE_LAST_LOGIN", cast=bool, default=True),
}


# -------------------------
# EMAIL_SERVICE
# -------------------------
EMAIL_BACKEND = config("EMAIL_BACKEND")
EMAIL_HOST = config("EMAIL_HOST")
EMAIL_PORT = config("EMAIL_PORT", cast=int, default=587)
EMAIL_USE_TLS = config("EMAIL_USE_TLS", cast=bool, default=True)
EMAIL_USE_SSL = config("EMAIL_USE_SSL", cast=bool, default=False)
EMAIL_HOST_USER = config("EMAIL_HOST_USER")
DEFAULT_FROM_EMAIL = config("DEFAULT_FROM_EMAIL")
EMAIL_HOST_PASSWORD = config("EMAIL_HOST_PASSWORD")
EMAIL_TOKEN_RESET_TIMEOUT= 120






INTERNAL_IPS = [
    # ...
    "127.0.0.1",
    # ...
]


# -------------------------
# CORS OPTIONS
# -------------------------
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True

CSRF_TRUSTED_ORIGINS = config("CSRF_TRUSTED_ORIGINS", cast=Csv(), default="")

# CORS_ALLOWED_ORIGINS = [
    # "http://localhost:5173",
#     "http://localhost:8080",
#     "http://localhost:8000",
# ]

CORS_ALLOW_METHODS = (
    "DELETE",
    "GET",
    "OPTIONS",
    "PATCH",
    "POST",
    "PUT",
)


# -------------------------
# CELERY
# -------------------------
CELERY_BROKER_URL = config("CELERY_BROKER_URL", default="redis://127.0.0.1:6379/1")
CELERY_RESULT_BACKEND = config("CELERY_RESULT_BACKEND", default="redis://127.0.0.1:6379/1")