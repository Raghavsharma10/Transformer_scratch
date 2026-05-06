def NegateQueryFilter(es_query):  # noqa
    """
    Return a filter removing the contents of the provided query.
    """
    query = es_query.to_dict().get("query", {})
    filtered = query.get("filtered", {})
    negated_filter = filtered.get("filter", {})
    return Not(**negated_filter)