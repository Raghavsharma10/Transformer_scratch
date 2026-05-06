def list_subadres_adapter(obj, request):
    """
    Adapter for rendering a list of
    :class:`crabpy.gateway.crab.Subadres` to json.
    """
    return {
        'id': obj.id,
        'subadres': obj.subadres,
        'status': {
            'id': obj.status.id,
            'naam': obj.status.naam,
            'definitie': obj.status.definitie
        }
    }