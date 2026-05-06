def search(query, model):
    """ Performs a search query and returns the object ids """
    query = query.strip()
    LOGGER.debug(query)
    sqs = SearchQuerySet()
    results = sqs.raw_search('{}*'.format(query)).models(model)
    if not results:
        results = sqs.raw_search('*{}'.format(query)).models(model)
    if not results:
        results = sqs.raw_search('*{}*'.format(query)).models(model)

    return [o.pk for o in results]