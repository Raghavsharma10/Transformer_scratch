def post_save_update_cache(sender, instance, created, raw, **kwargs):
    """Update the cache when an instance is created or modified."""
    if raw:
        return
    name = sender.__name__
    if name in cached_model_names:
        delay_cache = getattr(instance, '_delay_cache', False)
        if not delay_cache:
            from .tasks import update_cache_for_instance
            update_cache_for_instance(name, instance.pk, instance)