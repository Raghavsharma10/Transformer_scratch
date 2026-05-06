def item_deelgemeente_adapter(obj, request):
    """
    Adapter for rendering a object of
    :class:`crabpy.gateway.crab.Deelgemeente` to json.
    """
    return {
        'id': obj.id,
        'naam': obj.naam,
        'gemeente': {
            'id': obj.gemeente.id,
            'naam': obj.gemeente.naam
        }
    }