def item_gewest_adapter(obj, request):
    """
    Adapter for rendering an object of
    :class:`crabpy.gateway.crab.Gewest` to json.
    """
    return {
        'id': obj.id,
        'namen': obj._namen,
        'centroid': obj.centroid,
        'bounding_box': obj.bounding_box
    }