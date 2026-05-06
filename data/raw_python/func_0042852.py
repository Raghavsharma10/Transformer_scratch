def get_decoders_by_path(query):
    """
    RETURN MAP FROM QUERY PATH TO LIST OF DECODER ARRAYS

    :param query:
    :return:
    """
    schema = query.frum.schema
    output = Data()

    if query.edges:
        if query.sort and query.format != "cube":
            # REORDER EDGES/GROUPBY TO MATCH THE SORT
            query.edges = sort_edges(query, "edges")
    elif query.groupby:
        if query.sort and query.format != "cube":
            query.groupby = sort_edges(query, "groupby")

    for edge in wrap(coalesce(query.edges, query.groupby, [])):
        limit = coalesce(edge.domain.limit, query.limit, DEFAULT_LIMIT)
        if edge.value != None and not edge.value is NULL:
            edge = edge.copy()
            vars_ = edge.value.vars()
            for v in vars_:
                if not schema.leaves(v.var):
                    Log.error("{{var}} does not exist in schema", var=v)
        elif edge.range:
            vars_ = edge.range.min.vars() | edge.range.max.vars()
            for v in vars_:
                if not schema[v.var]:
                    Log.error("{{var}} does not exist in schema", var=v)
        elif edge.domain.dimension:
            vars_ = edge.domain.dimension.fields
            edge.domain.dimension = edge.domain.dimension.copy()
            edge.domain.dimension.fields = [schema[v].es_column for v in vars_]
        elif all(edge.domain.partitions.where):
            vars_ = set()
            for p in edge.domain.partitions:
                vars_ |= p.where.vars()

        vars_ |= edge.value.vars()
        depths = set(c.nested_path[0] for v in vars_ for c in schema.leaves(v.var))
        if not depths:
            Log.error(
                "Do not know of column {{column}}",
                column=unwraplist([v for v in vars_ if schema[v] == None])
            )
        if len(depths) > 1:
            Log.error("expression {{expr|quote}} spans tables, can not handle", expr=edge.value)

        decoder = AggsDecoder(edge, query, limit)
        output[literal_field(first(depths))] += [decoder]
    return output