from django.contrib import admin

# Register your models here.
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User,UserProfile

@admin.register(User)
class UserAdmin(BaseUserAdmin):
    ordering = ("-date_joined",)
    list_display = ("id", "email", "full_name", "is_email_verified", "is_staff", "date_joined")
    list_filter = ("is_staff", "is_superuser", "is_email_verified")
    fieldsets = (
        (None, {"fields": ("email", "password")}),
        ("Personal info", {"fields": ("full_name",)}),
        ("Permissions", {"fields": ("is_active","is_email_verified","is_staff","is_superuser","role")}),
        ("Important dates", {"fields": ("last_login","date_joined")}),
    )
    add_fieldsets = ((None, {"classes": ("wide",), "fields": ("email","password1","password2")}),)
    search_fields = ("email", "full_name")



@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "first_name", "last_name", "phone_number", "date_of_birth")
    search_fields = ("user__username", "first_name", "last_name", "phone_number")
    list_filter = ("date_of_birth",)