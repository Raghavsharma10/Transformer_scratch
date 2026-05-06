def get_record_collections(record, matcher):
    """Return list of collections to which record belongs to.

    :param record: Record instance.
    :param matcher: Function used to check if a record belongs to a collection.
    :return: list of collection names.
    """
    collections = current_collections.collections
    if collections is None:
        # build collections cache
        collections = current_collections.collections = dict(_build_cache())

    output = set()

    for collections in matcher(collections, record):
        output |= collections

    return list(output)