def apply(query, collection=None):
    """Enhance the query restricting not permitted collections.

    Get the permitted restricted collection for the current user from the
    user_info object and all the restriced collections from the
    restricted_collection_cache.
    """
    if not collection:
        return query
    result_tree = create_collection_query(collection)
    return AndOp(query, result_tree)