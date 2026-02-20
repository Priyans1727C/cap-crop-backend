from django.db import models

# Create your models here.
from django.utils import timezone
from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager,PermissionsMixin
from django.db.models.functions import Lower
from django.core.validators import FileExtensionValidator
# Create your models here.

class UserManager(BaseUserManager):
    def create_user(self,email,password=None,role="CUSTOMER",**extra):
        if not email:
            raise ValueError("Email is required")
        email = self.normalize_email(email)
        user = self.model(email=email,role=role,**extra)
        user.set_password(password) 
        user.save(using=self._db)
        return user
    def create_superuser(self,email,password,**extra):
        extra.setdefault("role","ADMIN")
        extra.setdefault("is_staff",True)
        extra.setdefault("is_superuser",True)
        if not password: raise ValueError("Superuser need password")
        return self.create_user(email,password,**extra)
    
        


class User(AbstractBaseUser,PermissionsMixin):
    class Roles(models.TextChoices):
        ADMIN = "ADMIN", "Admin"
        STAFF = "STAFF","Staff"
        CUSTOMER = "CUSTOMER","Customer"
    
    role = models.CharField(choices=Roles,default=Roles.CUSTOMER,max_length=20)
    email = models.EmailField(unique=True,db_index=True)
    full_name = models.CharField(max_length=150,blank=True)
    is_email_verified = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)  # Django admin access
    date_joined = models.DateTimeField(default=timezone.now)
    last_login = models.DateTimeField(null=True, blank=True)
    
    USERNAME_FIELD = "email"
    EMAIL_FIELD = "email"
    REQUIRED_FIELDS = []
    objects = UserManager()
    
    class Meta:
        indexes = [models.Index(Lower("email"),name="idx_user_email_ci")]
        constraints = [
            models.UniqueConstraint(
                Lower("email"),
                name="uniq_user_email_ci",
            )
        ]
    
    def __str__(self):
        return self.full_name or self.email
    
    
    @property
    def is_admin(self): return self.role==self.Roles.ADMIN 
    @property
    def is_staff_member(self): return self.role==self.Roles.STAFF
    @property
    def is_customer(self): return self.role==self.Roles.CUSTOMER 
    
    
def profile_image_upload_to(instance,filename):
    return f"profiles/{instance.user.email}-{filename}"
    
class UserProfile(models.Model):    
    user = models.OneToOneField(User,on_delete=models.CASCADE, to_field='email' ,primary_key=True, related_name="profile")
    first_name = models.CharField(max_length=50, blank=True)
    last_name = models.CharField(max_length=50, blank=True)
    phone_number = models.CharField(max_length=15, blank=True, null=True)  
    address = models.TextField(max_length=500, blank=True)
    date_of_birth = models.DateField(blank=True, null=True)
    image = models.ImageField(upload_to=profile_image_upload_to,blank=True,null=True,
                              validators=[FileExtensionValidator(allowed_extensions=["jpg","jpeg","png","webp"])])
    

    def __str__(self):
        return f"{self.first_name or self.user.email}'s Profile"
    