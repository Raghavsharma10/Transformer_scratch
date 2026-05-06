def es_query_proto(path, selects, wheres, schema):
    """
    RETURN TEMPLATE AND PATH-TO-FILTER AS A 2-TUPLE
    :param path: THE NESTED PATH (NOT INCLUDING TABLE NAME)
    :param wheres: MAP FROM path TO LIST OF WHERE CONDITIONS
    :return: (es_query, filters_map) TUPLE
    """
    output = None
    last_where = MATCH_ALL
    for p in reversed(sorted( wheres.keys() | set(selects.keys()))):
        where = wheres.get(p)
        select = selects.get(p)

        if where:
            where = AndOp(where).partial_eval().to_esfilter(schema)
            if output:
                where = es_or([es_and([output, where]), where])
        else:
            if output:
                if last_where is MATCH_ALL:
                    where = es_or([output, MATCH_ALL])
                else:
                    where = output
            else:
                where = MATCH_ALL

        if p == ".":
            output = set_default(
                {
                    "from": 0,
                    "size": 0,
                    "sort": [],
                    "query": where
                },
                select.to_es()
            )
        else:
            output = {"nested": {
                "path": p,
                "inner_hits": set_default({"size": 100000}, select.to_es()) if select else None,
                "query": where
            }}

        last_where = where
    return output