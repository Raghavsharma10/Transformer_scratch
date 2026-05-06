def prune_index(index):
    """Remove all orphaned documents from an index.

    This function works by scanning the remote index, and in each returned
    batch of documents looking up whether they appear in the default index
    queryset. If they don't (they've been deleted, or no longer fit the qs
    filters) then they are deleted from the index. The deletion is done in
    one hit after the entire remote index has been scanned.

    The elasticsearch.helpers.scan function returns each document one at a
    time, so this function can swamp the database with SELECT requests.

    Please use sparingly.

    Returns a list of ids of all the objects deleted.

    """
    logger.info("Pruning missing objects from index '%s'", index)
    prunes = []
    responses = []
    client = get_client()
    for model in get_index_models(index):
        for hit in scan_index(index, model):
            obj = _prune_hit(hit, model)
            if obj:
                prunes.append(obj)
        logger.info(
            "Found %s objects of type '%s' for deletion from '%s'.",
            len(prunes),
            model,
            index,
        )
        if len(prunes) > 0:
            actions = bulk_actions(prunes, index, "delete")
            response = helpers.bulk(
                client, actions, chunk_size=get_setting("chunk_size")
            )
            responses.append(response)
    return responses