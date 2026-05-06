def scan_index(index, model):
    """
    Yield all documents of model type in an index.

    This function calls the elasticsearch.helpers.scan function,
    and yields all the documents in the index that match the doc_type
    produced by a specific Django model.

    Args:
        index: string, the name of the index to scan, must be a configured
            index as returned from settings.get_index_names.
        model: a Django model type, used to filter the the documents that
            are scanned.

    Yields each document of type model in index, one at a time.

    """
    # see https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-type-query.html
    query = {"query": {"type": {"value": model._meta.model_name}}}
    client = get_client()
    for hit in helpers.scan(client, index=index, query=query):
        yield hit