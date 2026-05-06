def es_query_template(path):
    """
    RETURN TEMPLATE AND PATH-TO-FILTER AS A 2-TUPLE
    :param path: THE NESTED PATH (NOT INCLUDING TABLE NAME)
    :return: (es_query, es_filters) TUPLE
    """

    if not is_text(path):
        Log.error("expecting path to be a string")

    if path != ".":
        f0 = {}
        f1 = {}
        output = wrap({
            "query": es_and([
                f0,
                {"nested": {
                    "path": path,
                    "query": f1,
                    "inner_hits": {"size": 100000}
                }}
            ]),
            "from": 0,
            "size": 0,
            "sort": []
        })
        return output, wrap([f0, f1])
    else:
        f0 = {}
        output = wrap({
            "query": es_and([f0]),
            "from": 0,
            "size": 0,
            "sort": []
        })
        return output, wrap([f0])