def _facet_counts(items):
    """Returns facet counts as dict.

    Given the `items()` on the raw dictionary from Elasticsearch this processes
    it and returns the counts keyed on the facet name provided in the original
    query.

    """
    facets = {}
    for name, data in items:
        facets[name] = FacetResult(name, data)
    return facets