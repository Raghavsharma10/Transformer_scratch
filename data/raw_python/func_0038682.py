def custom_search_model(model, query, preview=False, published=False,
                        id_field="id", sort_pinned=True, field_map={}):
    """Filter a model with the given filter.

    `field_map` translates incoming field names to the appropriate ES names.
    """
    if preview:
        func = preview_filter_from_query
    else:
        func = filter_from_query
    f = func(query, id_field=id_field, field_map=field_map)
    # filter by published
    if published:
        if f:
            f &= Range(published={"lte": timezone.now()})
        else:
            f = Range(published={"lte": timezone.now()})

    qs = model.search_objects.search(published=False)
    if f:
        qs = qs.filter(f)

    # possibly include a text query
    if query.get("query"):
        qs = qs.query("match", _all=query["query"])
    # set up pinned ids
    pinned_ids = query.get("pinned_ids")
    if pinned_ids and sort_pinned:

        pinned_query = es_query.FunctionScore(
            boost_mode="multiply",
            functions=[{
                "filter": Terms(id=pinned_ids),
                "weight": 2
            }]
        )

        qs = qs.query(pinned_query)
        qs = qs.sort("_score", "-published")
    else:
        qs = qs.sort("-published")
    return qs