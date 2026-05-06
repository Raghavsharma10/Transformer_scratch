def item_provincie_adapter(obj, request):
    """
    Adapter for rendering a object of
    :class:`crabpy.gateway.crab.Provincie` to json.
    """
    return {
        'niscode': obj.niscode,
        'naam': obj.naam,
        'gewest': {
            'id': obj.gewest.id,
            'naam': obj.gewest.naam
        }
    }