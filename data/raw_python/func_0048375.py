def item_adrespositie_adapter(obj, request):
    """
    Adapter for rendering an item of
    :class:`crabpy.gateway.Adrespositie` to json.
    """
    return {
        'id': obj.id,
        'herkomst': {
            'id': obj.herkomst.id,
            'naam': obj.herkomst.naam,
            'definitie': obj.herkomst.definitie
        },
        'geometrie': obj.geometrie,
        'aard': {
            'id': obj.aard.id,
            'naam': obj.aard.naam,
            'definitie': obj.aard.definitie
        },
        'metadata': {
            'begin_tijd': obj.metadata.begin_tijd,
            'begin_datum': obj.metadata.begin_datum,
            'begin_bewerking': {
                'id': obj.metadata.begin_bewerking.id,
                'naam': obj.metadata.begin_bewerking.naam,
                'definitie': obj.metadata.begin_bewerking.definitie
            },
            'begin_organisatie': {
                'id': obj.metadata.begin_organisatie.id,
                'naam': obj.metadata.begin_organisatie.naam,
                'definitie': obj.metadata.begin_organisatie.definitie
            }
        }
    }