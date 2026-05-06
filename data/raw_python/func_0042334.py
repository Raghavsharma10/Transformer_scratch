def run(query, container=Null):
    """
    THIS FUNCTION IS SIMPLY SWITCHING BASED ON THE query["from"] CONTAINER,
    BUT IT IS ALSO PROCESSING A list CONTAINER; SEPARATE TO A ListContainer
    """
    if container == None:
        container = wrap(query)["from"]
        query_op = QueryOp.wrap(query, container=container, namespace=container.schema)
    else:
        query_op = QueryOp.wrap(query, container, container.namespace)

    if container == None:
        from jx_python.containers.list_usingPythonList import DUAL

        return DUAL.query(query_op)
    elif isinstance(container, Container):
        return container.query(query_op)
    elif is_many(container):
        container = wrap(list(container))
    elif isinstance(container, Cube):
        if is_aggs(query_op):
            return cube_aggs(container, query_op)
    elif is_op(container, QueryOp):
        container = run(container)
    elif is_data(container):
        query = container
        container = query["from"]
        container = run(QueryOp.wrap(query, container, container.namespace), container)
    else:
        Log.error(
            "Do not know how to handle {{type}}", type=container.__class__.__name__
        )

    if is_aggs(query_op):
        container = list_aggs(container, query_op)
    else:  # SETOP
        if query_op.where is not TRUE:
            container = filter(container, query_op.where)

        if query_op.sort:
            container = sort(container, query_op.sort, already_normalized=True)

        if query_op.select:
            container = select(container, query_op.select)

    if query_op.window:
        if isinstance(container, Cube):
            container = list(container.values())

        for param in query_op.window:
            window(container, param)

    # AT THIS POINT frum IS IN LIST FORMAT, NOW PACKAGE RESULT
    if query_op.format == "cube":
        container = convert.list2cube(container)
    elif query_op.format == "table":
        container = convert.list2table(container)
        container.meta.format = "table"
    else:
        container = wrap({"meta": {"format": "list"}, "data": container})

    return container