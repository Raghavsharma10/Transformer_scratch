def execute_search(
    search,
    search_terms="",
    user=None,
    reference="",
    save=True,
    query_type=SearchQuery.QUERY_TYPE_SEARCH,
):
    """
    Create a new SearchQuery instance and execute a search against ES.

    Args:
        search: elasticsearch.search.Search object, that internally contains
            the connection and query; this is the query that is executed. All
            we are doing is logging the input and parsing the output.
        search_terms: raw end user search terms input - what they typed into the search
            box.
        user: Django User object, the person making the query - used for logging
            purposes. Can be null.
        reference: string, can be anything you like, used for identification,
            grouping purposes.
        save: bool, if True then save the new object immediately, can be
            overridden to False to prevent logging absolutely everything.
            Defaults to True
        query_type: string, used to determine whether to run a search query or
            a count query (returns hit count, but no results).

    """
    start = time.time()
    if query_type == SearchQuery.QUERY_TYPE_SEARCH:
        response = search.execute()
        hits = [h.meta.to_dict() for h in response.hits]
        total_hits = response.hits.total
    elif query_type == SearchQuery.QUERY_TYPE_COUNT:
        response = total_hits = search.count()
        hits = []
    else:
        raise ValueError(f"Invalid SearchQuery.query_type value: '{query_type}'")
    duration = time.time() - start
    search_query = SearchQuery(
        user=user,
        search_terms=search_terms,
        index=", ".join(search._index or ["_all"])[:100],  # field length restriction
        query=search.to_dict(),
        query_type=query_type,
        hits=hits,
        total_hits=total_hits,
        reference=reference or "",
        executed_at=tz_now(),
        duration=duration,
    )
    search_query.response = response
    return search_query.save() if save else search_query