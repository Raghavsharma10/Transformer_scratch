def unindex_layer(self, layer_id, use_cache=False):
    """
    Remove the index for a layer in the search backend.
    If cache is set, append it to the list of removed layers, if it isn't send the transaction right away.
    """

    from hypermap.aggregator.models import Layer
    layer = Layer.objects.get(id=layer_id)

    if use_cache:
        LOGGER.debug('Caching layer with id %s for being removed from search engine' % layer.id)
        deleted_layers = cache.get('deleted_layers')
        if deleted_layers is None:
            deleted_layers = set([layer.id])
        else:
            deleted_layers.add(layer.id)
        cache.set('deleted_layers', deleted_layers)
        return

    if SEARCH_TYPE == 'solr':
        from hypermap.aggregator.solr import SolrHypermap
        LOGGER.debug('Removing layer %s from solr' % layer.id)
        try:
            solrobject = SolrHypermap()
            solrobject.remove_layer(layer.uuid)
        except Exception:
            LOGGER.error('Layer NOT correctly removed from Solr')
    elif SEARCH_TYPE == 'elasticsearch':
        # TODO implement me
        pass