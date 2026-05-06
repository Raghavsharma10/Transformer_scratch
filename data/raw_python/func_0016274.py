def _prune_hit(hit, model):
    """
    Check whether a document should be pruned.

    This method uses the SearchDocumentManagerMixin.in_search_queryset method
    to determine whether a 'hit' (search document) should be pruned from an index,
    and if so it returns the hit as a Django object(id=hit_id).

    Args:
        hit: dict object the represents a document as returned from the scan_index
            function. (Contains object id and index.)
        model: the Django model (not object) from which the document was derived.
            Used to get the correct model manager and bulk action.

    Returns:
        an object of type model, with id=hit_id. NB this is not the object
        itself, which by definition may not exist in the underlying database,
        but a temporary object with the document id - which is enough to create
        a 'delete' action.

    """
    hit_id = hit["_id"]
    hit_index = hit["_index"]
    if model.objects.in_search_queryset(hit_id, index=hit_index):
        logger.debug(
            "%s with id=%s exists in the '%s' index queryset.", model, hit_id, hit_index
        )
        return None
    else:
        logger.debug(
            "%s with id=%s does not exist in the '%s' index queryset and will be pruned.",
            model,
            hit_id,
            hit_index,
        )
        # we don't need the full obj for a delete action, just the id.
        # (the object itself may not even exist.)
        return model(pk=hit_id)