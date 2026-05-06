def service_post_save(instance, *args, **kwargs):
    """
    Used to do a service full check when saving it.
    """

    # check service
    if instance.is_monitored and settings.REGISTRY_SKIP_CELERY:
        check_service(instance.id)
    elif instance.is_monitored:
        check_service.delay(instance.id)