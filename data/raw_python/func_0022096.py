def update_last_wm_layers(self, service_id, num_layers=10):
    """
    Update and index the last added and deleted layers (num_layers) in WorldMap service.
    """
    from hypermap.aggregator.models import Service

    LOGGER.debug(
        'Updating the index the last %s added and %s deleted layers in WorldMap service'
        % (num_layers, num_layers)
                )
    service = Service.objects.get(id=service_id)
    # TODO raise error if service type is not WM type
    if service.type == 'Hypermap:WorldMapLegacy':
        from hypermap.aggregator.models import update_layers_wm_legacy as update_layers_wm
    if service.type == 'Hypermap:WorldMap':
        from hypermap.aggregator.models import update_layers_geonode_wm as update_layers_wm

    update_layers_wm(service, num_layers)

    # Remove in search engine last num_layers that were deleted
    LOGGER.debug('Removing the index for the last %s deleted layers' % num_layers)
    layer_to_unindex = service.layer_set.filter(was_deleted=True).order_by('-last_updated')[0:num_layers]
    for layer in layer_to_unindex:
        if not settings.REGISTRY_SKIP_CELERY:
            unindex_layer(layer.id, use_cache=True)
        else:
            unindex_layer(layer.id)

    # Add/Update in search engine last num_layers that were added
    LOGGER.debug('Adding/Updating the index for the last %s added layers' % num_layers)
    layer_to_index = service.layer_set.filter(was_deleted=False).order_by('-last_updated')[0:num_layers]
    for layer in layer_to_index:
        if not settings.REGISTRY_SKIP_CELERY:
            index_layer(layer.id, use_cache=True)
        else:
            index_layer(layer.id)