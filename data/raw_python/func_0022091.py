def index_service(self, service_id):
    """
    Index a service in search engine.
    """

    from hypermap.aggregator.models import Service
    service = Service.objects.get(id=service_id)

    if not service.is_valid:
        LOGGER.debug('Not indexing service with id %s in search engine as it is not valid' % service.id)
        return

    LOGGER.debug('Indexing service %s' % service.id)
    layer_to_process = service.layer_set.all()

    for layer in layer_to_process:
        if not settings.REGISTRY_SKIP_CELERY:
            index_layer(layer.id, use_cache=True)
        else:
            index_layer(layer.id)