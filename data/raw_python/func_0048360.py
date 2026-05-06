def list_straten_adapter(obj, request):
    """
    Adapter for rendering a list of
    :class:`crabpy.gateway.crab.Straat` to json.
    """
    return {
        'id': obj.id,
        'label': obj.label,
        'status': {
            'id': obj.status.id,
            'naam': obj.status.naam,
            'definitie': obj.status.definitie
        },
    }