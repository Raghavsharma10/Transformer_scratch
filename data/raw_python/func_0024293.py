def search_unit(current):
    """
        Search on units for subscribing it's users to a channel

        .. code-block:: python

            #  request:
                {
                'view':'_zops_search_unit',
                'query': string,
                }

            #  response:
                {
                'results': [('name', 'key'), ],
                'status': 'OK',
                'code': 200
                }
    """
    current.output = {
        'results': [],
        'status': 'OK',
        'code': 201
    }
    for user in UnitModel(current).objects.search_on(*settings.MESSAGING_UNIT_SEARCH_FIELDS,
                                                     contains=current.input['query']):
        current.output['results'].append((user.name, user.key))