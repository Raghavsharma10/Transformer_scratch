def issue_post_delete(instance, *args, **kwargs):
    """
    Used to do reindex layers/services when a issue is removed form them.
    """
    LOGGER.debug('Re-adding layer/service to search engine index')
    if isinstance(instance.content_object, Service):
        if not settings.REGISTRY_SKIP_CELERY:
            index_service.delay(instance.content_object.id)
        else:
            index_service(instance.content_object.id)
    else:
        if not settings.REGISTRY_SKIP_CELERY:
            index_layer.delay(instance.content_object.id)
        else:
            index_layer(instance.content_object.id)