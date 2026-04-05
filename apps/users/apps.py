from django.apps import AppConfig


class UsersConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.users'

    def ready(self):
        try:
            from . import signals
        except Exception:
            print("Field to import apps.users.signals.")
        