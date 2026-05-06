def index_layer(self, layer_id, use_cache=False):
    """
    Index a layer in the search backend.
    If cache is set, append it to the list, if it isn't send the transaction right away.
    cache needs memcached to be available.
    """

    from hypermap.aggregator.models import Layer
    layer = Layer.objects.get(id=layer_id)

    if not layer.is_valid:
        LOGGER.debug('Not indexing or removing layer with id %s in search engine as it is not valid' % layer.id)
        unindex_layer(layer.id, use_cache)
        return

    if layer.was_deleted:
        LOGGER.debug('Not indexing or removing layer with id %s in search engine as was_deleted is true' % layer.id)
        unindex_layer(layer.id, use_cache)
        return

    # 1. if we use cache
    if use_cache:
        LOGGER.debug('Caching layer with id %s for syncing with search engine' % layer.id)
        layers = cache.get('layers')
        if layers is None:
            layers = set([layer.id])
        else:
            layers.add(layer.id)
        cache.set('layers', layers)
        return

    # 2. if we don't use cache
    # TODO: Make this function more DRY
    # by abstracting the common bits.
    if SEARCH_TYPE == 'solr':
        from hypermap.aggregator.solr import SolrHypermap
        LOGGER.debug('Syncing layer %s to solr' % layer.name)
        solrobject = SolrHypermap()
        success, message = solrobject.layer_to_solr(layer)
        # update the error message if using celery
        if not settings.REGISTRY_SKIP_CELERY:
            if not success:
                self.update_state(
                    state=states.FAILURE,
                    meta=message
                    )
                raise Ignore()
    elif SEARCH_TYPE == 'elasticsearch':
        from hypermap.aggregator.elasticsearch_client import ESHypermap
        LOGGER.debug('Syncing layer %s to es' % layer.name)
        esobject = ESHypermap()
        success, message = esobject.layer_to_es(layer)
        # update the error message if using celery
        if not settings.REGISTRY_SKIP_CELERY:
            if not success:
                self.update_state(
                    state=states.FAILURE,
                    meta=message
                    )
                raise Ignore()