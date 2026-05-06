def list_adresposities_adapter(obj, request):
    """
    Adapter for rendering a list of
    :class:`crabpy.gateway.crab.Adrespositie` to json.
    """
    return {
        'id': obj.id,
        'herkomst': {
            'id': obj.herkomst.id,
            'naam': obj.herkomst.naam,
            'definitie': obj.herkomst.definitie
        }
    }