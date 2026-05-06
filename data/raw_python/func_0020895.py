def _create_percolator_mapping(index, doc_type):
    """Update mappings with the percolator field.

    .. note::

        This is only needed from ElasticSearch v5 onwards, because percolators
        are now just a special type of field inside mappings.
    """
    if ES_VERSION[0] >= 5:
        current_search_client.indices.put_mapping(
            index=index, doc_type=doc_type,
            body=PERCOLATOR_MAPPING, ignore=[400, 404])