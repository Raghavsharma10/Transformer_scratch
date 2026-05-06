def item_wegobject_adapter(obj, request):
    """
    Adapter for rendering a list of
    :class:`crabpy.gateway.Wegobject` to json.
    """
    return {
        'id': obj.id,
        'aard': {
            'id': obj.aard.id,
            'naam': obj.aard.naam,
            'definitie': obj.aard.definitie
        },
        'centroid': obj.centroid,
        'bounding_box': obj.bounding_box,
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