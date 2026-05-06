def item_gebouw_adapter(obj, request):
    """
    Adapter for rendering an object of
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
        },
        'geometriemethode': {
            'id': obj.methode.id,
            'naam': obj.methode.naam,
            'definitie': obj.methode.definitie
        },
        'geometrie': obj.geometrie,
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