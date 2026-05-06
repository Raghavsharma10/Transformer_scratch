def _normalize_group(edge, dim_index, limit, schema=None):
    """
    :param edge: Not normalized groupby
    :param dim_index: Dimensions are ordered; this is this groupby's index into that order
    :param schema: for context
    :return: a normalized groupby
    """
    if is_text(edge):
        if edge.endswith(".*"):
            prefix = edge[:-2]
            if schema:
                output = wrap([
                    {  # BECASUE THIS IS A GROUPBY, EARLY SPLIT INTO LEAVES WORKS JUST FINE
                        "name": concat_field(prefix, literal_field(relative_field(untype_path(c.name), prefix))),
                        "put": {"name": literal_field(untype_path(c.name))},
                        "value": jx_expression(c.es_column, schema=schema),
                        "allowNulls": True,
                        "domain": {"type": "default"}
                    }
                    for c in schema.leaves(prefix)
                ])
                return output
            else:
                return wrap([{
                    "name": untype_path(prefix),
                    "put": {"name": literal_field(untype_path(prefix))},
                    "value": LeavesOp(Variable(prefix)),
                    "allowNulls": True,
                    "dim":dim_index,
                    "domain": {"type": "default"}
                }])

        return wrap([{
            "name": edge,
            "value": jx_expression(edge, schema=schema),
            "allowNulls": True,
            "dim": dim_index,
            "domain": Domain(type="default", limit=limit)
        }])
    else:
        edge = wrap(edge)
        if (edge.domain and edge.domain.type != "default") or edge.allowNulls != None:
            Log.error("groupby does not accept complicated domains")

        if not edge.name and not is_text(edge.value):
            Log.error("You must name compound edges: {{edge}}",  edge= edge)

        return wrap([{
            "name": coalesce(edge.name, edge.value),
            "value": jx_expression(edge.value, schema=schema),
            "allowNulls": True,
            "dim":dim_index,
            "domain": {"type": "default"}
        }])