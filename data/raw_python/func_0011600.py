def post_delete_update_cache(sender, instance, **kwargs):
    """Update the cache when an instance is deleted."""
    name = sender.__name__
    if name in cached_model_names:
        from .tasks import update_cache_for_instance
        update_cache_for_instance(name, instance.pk, instance)