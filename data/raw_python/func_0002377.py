def treenav_save_other_object_handler(sender, instance, created, **kwargs):
    """
    This signal attempts to update the HREF of any menu items that point to
    another model object, when that objects is saved.
    """
    # import here so models don't get loaded during app loading
    from django.contrib.contenttypes.models import ContentType
    from .models import MenuItem
    cache_key = 'django-treenav-menumodels'
    if sender == MenuItem:
        cache.delete(cache_key)
    menu_models = cache.get(cache_key)
    if not menu_models:
        menu_models = []
        for menu_item in MenuItem.objects.exclude(content_type__isnull=True):
            menu_models.append(menu_item.content_type.model_class())
        cache.set(cache_key, menu_models)
    # only attempt to update MenuItem if sender is known to be referenced
    if sender in menu_models:
        ct = ContentType.objects.get_for_model(sender)
        items = MenuItem.objects.filter(content_type=ct, object_id=instance.pk)
        for item in items:
            if item.href != instance.get_absolute_url():
                item.href = instance.get_absolute_url()
                item.save()