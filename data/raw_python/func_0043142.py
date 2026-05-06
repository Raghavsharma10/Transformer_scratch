def wrap(query, container, namespace):
        """
        NORMALIZE QUERY SO IT CAN STILL BE JSON
        """
        if is_op(query, QueryOp) or query == None:
            return query

        query = wrap(query)
        table = container.get_table(query['from'])
        schema = table.schema
        output = QueryOp(
            frum=table,
            format=query.format,
            limit=mo_math.min(MAX_LIMIT, coalesce(query.limit, DEFAULT_LIMIT))
        )

        if query.select or isinstance(query.select, (Mapping, list)):
            output.select = _normalize_selects(query.select, query.frum, schema=schema)
        else:
            if query.edges or query.groupby:
                output.select = DEFAULT_SELECT
            else:
                output.select = _normalize_selects(".", query.frum)

        if query.groupby and query.edges:
            Log.error("You can not use both the `groupby` and `edges` clauses in the same query!")
        elif query.edges:
            output.edges = _normalize_edges(query.edges, limit=output.limit, schema=schema)
            output.groupby = Null
        elif query.groupby:
            output.edges = Null
            output.groupby = _normalize_groupby(query.groupby, limit=output.limit, schema=schema)
        else:
            output.edges = Null
            output.groupby = Null

        output.where = _normalize_where(query.where, schema=schema)
        output.window = [_normalize_window(w) for w in listwrap(query.window)]
        output.having = None
        output.sort = _normalize_sort(query.sort)
        if not mo_math.is_integer(output.limit) or output.limit < 0:
            Log.error("Expecting limit >= 0")

        output.isLean = query.isLean

        return output