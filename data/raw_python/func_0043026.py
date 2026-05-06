def split_expression_by_path(
    where, schema, output=None, var_to_columns=None, lang=Language
):
    """
    :param where: EXPRESSION TO INSPECT
    :param schema: THE SCHEMA
    :param output: THE MAP FROM PATH TO EXPRESSION WE WANT UPDATED
    :param var_to_columns: MAP FROM EACH VARIABLE NAME TO THE DEPTH
    :return: output: A MAP FROM PATH TO EXPRESSION
    """
    if var_to_columns is None:
        var_to_columns = {v.var: schema.leaves(v.var) for v in where.vars()}
        output = wrap({schema.query_path[0]: []})
        if not var_to_columns:
            output["\\."] += [where]  # LEGIT EXPRESSIONS OF ZERO VARIABLES
            return output

    where_vars = where.vars()
    all_paths = set(c.nested_path[0] for v in where_vars for c in var_to_columns[v.var])

    if len(all_paths) == 0:
        output["\\."] += [where]
    elif len(all_paths) == 1:
        output[literal_field(first(all_paths))] += [
            where.map(
                {
                    v.var: c.es_column
                    for v in where.vars()
                    for c in var_to_columns[v.var]
                }
            )
        ]
    elif is_op(where, AndOp_):
        for w in where.terms:
            split_expression_by_path(w, schema, output, var_to_columns, lang=lang)
    else:
        Log.error("Can not handle complex where clause")

    return output