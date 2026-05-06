def addLayer(layer, loadInLegend=True):
    """
    Add one or several layers to the QGIS session and layer registry.
    :param layer: The layer object or list with layers  to add the QGIS layer registry and session.
    :param loadInLegend: True if this layer should be added to the legend.
    :return: The added layer
    """
    if not hasattr(layer, "__iter__"):
        layer = [layer]
    _layerreg.addMapLayers(layer, loadInLegend)
    return layer