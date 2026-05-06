def item_perceel_crab_adapter(obj, request):
    """
    Adapter for rendering an object of
    :class:`crabpy.gateway.crab.Perceel` to json.
    """
    return {
        'id': obj.id,
        'centroid': obj.centroid,
        'postadressen': obj.postadressen,
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