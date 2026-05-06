def _find_matching_collections_internally(collections, record):
    """Find matching collections with internal engine.

    :param collections: set of collections where search
    :param record: record to match
    """
    for name, data in iteritems(collections):
        if _build_query(data['query']).match(record):
            yield data['ancestors']
    raise StopIteration