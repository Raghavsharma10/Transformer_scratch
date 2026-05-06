def create_index(index):
    """Create an index and apply mapping if appropriate."""
    logger.info("Creating search index: '%s'", index)
    client = get_client()
    return client.indices.create(index=index, body=get_index_mapping(index))