def layerFromName(name):
    '''
    Returns the layer from the current project with the passed name
    Raises WrongLayerNameException if no layer with that name is found
    If several layers with that name exist, only the first one is returned
    '''
    layers =_layerreg.mapLayers().values()
    for layer in layers:
        if layer.name() == name:
            return layer
    raise WrongLayerNameException()