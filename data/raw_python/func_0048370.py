def item_huisnummer_adapter(obj, request):
    """
    Adapter for rendering an object of
    :class:`crabpy.gateway.crab.Huisnummer` to json.
    """
    return {
        'id': obj.id,
        'huisnummer': obj.huisnummer,
        'postadres': obj.postadres,
        'status': {
            'id': obj.status.id,
            'naam': obj.status.naam,
            'definitie': obj.status.definitie
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
        },
        'bounding_box': obj.bounding_box
    }