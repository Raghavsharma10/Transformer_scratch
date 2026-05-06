def index_cached_layers(self):
    """
    Index and unindex all layers in the Django cache (Index all layers who have been checked).
    """
    from hypermap.aggregator.models import Layer

    if SEARCH_TYPE == 'solr':
        from hypermap.aggregator.solr import SolrHypermap
        solrobject = SolrHypermap()
    else:
        from hypermap.aggregator.elasticsearch_client import ESHypermap
        from elasticsearch import helpers
        es_client = ESHypermap()

    layers_cache = cache.get('layers')
    deleted_layers_cache = cache.get('deleted_layers')

    # 1. added layers cache
    if layers_cache:
        layers_list = list(layers_cache)
        LOGGER.debug('There are %s layers in cache: %s' % (len(layers_list), layers_list))

        batch_size = settings.REGISTRY_SEARCH_BATCH_SIZE
        batch_lists = [layers_list[i:i+batch_size] for i in range(0, len(layers_list), batch_size)]

        for batch_list_ids in batch_lists:
            layers = Layer.objects.filter(id__in=batch_list_ids)

            if batch_size > len(layers):
                batch_size = len(layers)

            LOGGER.debug('Syncing %s/%s layers to %s: %s' % (batch_size, len(layers_cache), layers, SEARCH_TYPE))

            try:
                # SOLR
                if SEARCH_TYPE == 'solr':
                    success, layers_errors_ids = solrobject.layers_to_solr(layers)
                    if success:
                        # remove layers from cache here
                        layers_cache = layers_cache.difference(set(batch_list_ids))
                        LOGGER.debug('Removing layers with id %s from cache' % batch_list_ids)
                        cache.set('layers', layers_cache)
                # ES
                elif SEARCH_TYPE == 'elasticsearch':
                    with_bulk, success = True, False
                    layers_to_index = [es_client.layer_to_es(layer, with_bulk) for layer in layers]
                    message = helpers.bulk(es_client.es, layers_to_index)

                    # Check that all layers where indexed...if not, don't clear cache.
                    # TODO: Check why es does not index all layers at first.
                    len_indexed_layers = message[0]
                    if len_indexed_layers == len(layers):
                        LOGGER.debug('%d layers indexed successfully' % (len_indexed_layers))
                        success = True
                    if success:
                        # remove layers from cache here
                        layers_cache = layers_cache.difference(set(batch_list_ids))
                        cache.set('layers', layers_cache)
                else:
                    raise Exception("Incorrect SEARCH_TYPE=%s" % SEARCH_TYPE)
            except Exception as e:
                LOGGER.error('Layers were NOT indexed correctly')
                LOGGER.error(e, exc_info=True)
    else:
        LOGGER.debug('No cached layers to add in search engine.')

    # 2. deleted layers cache
    if deleted_layers_cache:
        layers_list = list(deleted_layers_cache)
        LOGGER.debug('There are %s layers in cache for deleting: %s' % (len(layers_list), layers_list))
        # TODO implement me: batch layer index deletion
        for layer_id in layers_list:
            # SOLR
            if SEARCH_TYPE == 'solr':
                if Layer.objects.filter(pk=layer_id).exists():
                    layer = Layer.objects.get(id=layer_id)
                    unindex_layer(layer.id, use_cache=False)
                    deleted_layers_cache = deleted_layers_cache.difference(set([layer_id]))
                    cache.set('deleted_layers', deleted_layers_cache)
            else:
                # TODO implement me
                raise NotImplementedError
    else:
        LOGGER.debug('No cached layers to remove in search engine.')