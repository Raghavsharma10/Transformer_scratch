def _normalize_edge(edge, dim_index, limit, schema=None):
    """
    :param edge: Not normalized edge
    :param dim_index: Dimensions are ordered; this is this edge's index into that order
    :param schema: for context
    :return: a normalized edge
    """
    if not _Column:
        _late_import()

    if not edge:
        Log.error("Edge has no value, or expression is empty")
    elif is_text(edge):
        if schema:
            leaves = unwraplist(list(schema.leaves(edge)))
            if not leaves or is_container(leaves):
                return [
                    Data(
                        name=edge,
                        value=jx_expression(edge, schema=schema),
                        allowNulls=True,
                        dim=dim_index,
                        domain=_normalize_domain(None, limit)
                    )
                ]
            elif isinstance(leaves, _Column):
                return [Data(
                    name=edge,
                    value=jx_expression(edge, schema=schema),
                    allowNulls=True,
                    dim=dim_index,
                    domain=_normalize_domain(domain=leaves, limit=limit, schema=schema)
                )]
            elif is_list(leaves.fields) and len(leaves.fields) == 1:
                return [Data(
                    name=leaves.name,
                    value=jx_expression(leaves.fields[0], schema=schema),
                    allowNulls=True,
                    dim=dim_index,
                    domain=leaves.getDomain()
                )]
            else:
                return [Data(
                    name=leaves.name,
                    allowNulls=True,
                    dim=dim_index,
                    domain=leaves.getDomain()
                )]
        else:
            return [
                Data(
                    name=edge,
                    value=jx_expression(edge, schema=schema),
                    allowNulls=True,
                    dim=dim_index,
                    domain=DefaultDomain()
                )
            ]
    else:
        edge = wrap(edge)
        if not edge.name and not is_text(edge.value):
            Log.error("You must name compound and complex edges: {{edge}}", edge=edge)

        if is_container(edge.value) and not edge.domain:
            # COMPLEX EDGE IS SHORT HAND
            domain = _normalize_domain(schema=schema)
            domain.dimension = Data(fields=edge.value)

            return [Data(
                name=edge.name,
                value=jx_expression(edge.value, schema=schema),
                allowNulls=bool(coalesce(edge.allowNulls, True)),
                dim=dim_index,
                domain=domain
            )]

        domain = _normalize_domain(edge.domain, schema=schema)

        return [Data(
            name=coalesce(edge.name, edge.value),
            value=jx_expression(edge.value, schema=schema),
            range=_normalize_range(edge.range),
            allowNulls=bool(coalesce(edge.allowNulls, True)),
            dim=dim_index,
            domain=domain
        )]