def layerFromSource(source):
    '''
    Returns the layer from the current project with the passed source
    Raises WrongLayerSourceException if no layer with that source is found
    '''
    layers =_layerreg.mapLayers().values()
    for layer in layers:
        if layer.source() == source:
            return layer
    raise WrongLayerSourceException()