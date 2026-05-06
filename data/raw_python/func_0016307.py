def mapLayers(name=None, types=None):
    """
    Return all the loaded layers.  Filters by name (optional) first and then type (optional)
    :param name: (optional) name of layer to return..
    :param type: (optional) The QgsMapLayer type of layer to return. Accepts a single value or a list of them
    :return: List of loaded layers. If name given will return all layers with matching name.
    """
    if types is not None and not isinstance(types, list):
        types = [types]
    layers = _layerreg.mapLayers().values()
    _layers = []
    if name or types:
        if name:
            _layers = [layer for layer in layers if re.match(name, layer.name())]
        if types:
            _layers += [layer for layer in layers if layer.type() in types]
        return _layers
    else:
        return layers