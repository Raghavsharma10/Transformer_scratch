def _find_matching_collections_externally(collections, record):
    """Find matching collections with percolator engine.

    :param collections: set of collections where search
    :param record: record to match
    """
    index, doc_type = RecordIndexer().record_to_index(record)
    body = {"doc": record.dumps()}
    results = current_search_client.percolate(
        index=index,
        doc_type=doc_type,
        allow_no_indices=True,
        ignore_unavailable=True,
        body=body
    )
    prefix_len = len('collection-')
    for match in results['matches']:
        collection_name = match['_id']
        if collection_name.startswith('collection-'):
            name = collection_name[prefix_len:]
            if name in collections:
                yield collections[name]['ancestors']
    raise StopIteration