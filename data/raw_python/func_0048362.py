def list_gebouwen_adapter(obj, request):
    """
    Adapter for rendering a list of
    :class:`crabpy.gateway.crab.Gebouw` to json.
    """
    return {
        'id': obj.id,
        'aard': {
            'id': obj.aard.id,
            'naam': obj.aard.naam,
            'definitie': obj.aard.definitie
        },
        'status': {
            'id': obj.status.id,
            'naam': obj.status.naam,
            'definitie': obj.status.definitie
        }
    }