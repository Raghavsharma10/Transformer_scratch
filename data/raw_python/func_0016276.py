def bulk_actions(objects, index, action):
    """
    Yield bulk api 'actions' from a collection of objects.

    The output from this method can be fed in to the bulk
    api helpers - each document returned by get_documents
    is decorated with the appropriate bulk api op_type.

    Args:
        objects: iterable (queryset, list, ...) of SearchDocumentMixin
            objects. If the objects passed in is a generator, then this
                function will yield the results rather than returning them.
        index: string, the name of the index to target - the index name
            is embedded into the return value and is used by the bulk api.
        action: string ['index' | 'update' | 'delete'] - this decides
            how the final document is formatted.

    """
    assert (
        index != "_all"
    ), "index arg must be a valid index name. '_all' is a reserved term."
    logger.info("Creating bulk '%s' actions for '%s'", action, index)
    for obj in objects:
        try:
            logger.debug("Appending '%s' action for '%r'", action, obj)
            yield obj.as_search_action(index=index, action=action)
        except Exception:
            logger.exception("Unable to create search action for %s", obj)