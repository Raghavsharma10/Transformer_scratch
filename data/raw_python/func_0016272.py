def delete_index(index):
    """Delete index entirely (removes all documents and mapping)."""
    logger.info("Deleting search index: '%s'", index)
    client = get_client()
    return client.indices.delete(index=index)