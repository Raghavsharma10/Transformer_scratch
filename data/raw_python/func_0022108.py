def layer_post_save(instance, *args, **kwargs):
    """
    Used to do a layer full check when saving it.
    """
    if instance.is_monitored and instance.service.is_monitored:  # index and monitor
        if not settings.REGISTRY_SKIP_CELERY:
            check_layer.delay(instance.id)
        else:
            check_layer(instance.id)
    else:  # just index
        index_layer(instance.id)