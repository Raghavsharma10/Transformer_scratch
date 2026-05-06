def index():
    """Query Elasticsearch using "collection" param in query string."""
    collection_names = request.values.getlist('collection')

    # Validation of collection names.
    collections = Collection.query
    if collection_names:
        collections = collections.filter(
            Collection.name.in_(collection_names))
    assert len(collection_names) == collections.count()

    response = search.client.search(
        body={
            'query': {
                'filtered': {
                    'filter': {
                        'terms': {
                            '_collections': collection_names
                        }
                    }
                }
            }
        }
    )
    return jsonify(**response)