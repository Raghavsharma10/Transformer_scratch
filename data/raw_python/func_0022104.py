def endpointlist_post_save(instance, *args, **kwargs):
    """
    Used to process the lines of the endpoint list.
    """
    with open(instance.upload.file.name, mode='rb') as f:
        lines = f.readlines()
    for url in lines:
        if len(url) > 255:
            LOGGER.debug('Skipping this endpoint, as it is more than 255 characters: %s' % url)
        else:
            if Endpoint.objects.filter(url=url, catalog=instance.catalog).count() == 0:
                endpoint = Endpoint(url=url, endpoint_list=instance)
                endpoint.catalog = instance.catalog
                endpoint.save()
    if not settings.REGISTRY_SKIP_CELERY:
        update_endpoints.delay(instance.id)
    else:
        update_endpoints(instance.id)