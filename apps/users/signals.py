import os
from django.db.models.signals import pre_save, post_delete
from django.dispatch import receiver
from django.conf import settings
from .models import UserProfile

@receiver(pre_save, sender=UserProfile)
def delete_old_image_on_update(sender,instance, **kwargs):
    if not instance.pk:
        return
    
    try:
        user = UserProfile.objects.get(pk =instance.pk)
    except UserProfile.DoesNotExist:
        return
    
    old_profile_image = user.image
    new_profile_image = instance.image
    
    if old_profile_image and (not new_profile_image or old_profile_image!=new_profile_image):
        if old_profile_image.storage.exists(old_profile_image.name):
            old_profile_image.delete(old_profile_image.name)
            
@receiver(post_delete, sender=UserProfile)
def delete_image_on_profile_delete(sender,instance,**kwargs):
    if instance.image:
        try:
            if instance.image.store.exist(instance.image.name):
                instance.image.store.delete(instance.image.name)
        except Exception:
            pass