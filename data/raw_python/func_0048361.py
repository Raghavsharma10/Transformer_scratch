def list_huisnummers_adapter(obj, request):
    """
    Adapter for rendering a list of
    :class:`crabpy.gateway.crab.Huisnummer` to json.
    """
    return {
        'id': obj.id,
        'status': {
            'id': obj.status.id,
            'naam': obj.status.naam,
            'definitie': obj.status.definitie
        },
        'label': obj.huisnummer
    }