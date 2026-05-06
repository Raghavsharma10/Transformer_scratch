def index_all_layers(self):
    """
    Index all layers in search engine.
    """
    from hypermap.aggregator.models import Layer

    if not settings.REGISTRY_SKIP_CELERY:
        layers_cache = set(Layer.objects.filter(is_valid=True).values_list('id', flat=True))
        deleted_layers_cache = set(Layer.objects.filter(is_valid=False).values_list('id', flat=True))
        cache.set('layers', layers_cache)
        cache.set('deleted_layers', deleted_layers_cache)
    else:
        for layer in Layer.objects.all():
            index_layer(layer.id)