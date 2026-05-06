def index():
    """Query Elasticsearch using Invenio query syntax."""
    page = request.values.get('page', 1, type=int)
    size = request.values.get('size', 2, type=int)
    search = ExampleSearch()[(page - 1) * size:page * size]
    if 'q' in request.values:
        search = search.query(QueryString(query=request.values.get('q')))

    search = search.sort(
        request.values.get('sort', 'title')
    )
    search = ExampleSearch.faceted_search(search=search)
    results = search.execute().to_dict()
    return jsonify({'hits': results.get('hits')})