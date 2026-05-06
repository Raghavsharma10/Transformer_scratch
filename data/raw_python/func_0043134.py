def _normalize_select(select, frum, schema=None):
    """
    :param select: ONE SELECT COLUMN
    :param frum: TABLE TO get_columns()
    :param schema: SCHEMA TO LOOKUP NAMES FOR DEFINITIONS
    :return: AN ARRAY OF SELECT COLUMNS
    """
    if not _Column:
        _late_import()

    if is_text(select):
        canonical = select = Data(value=select)
    else:
        select = wrap(select)
        canonical = select.copy()

    canonical.aggregate = coalesce(canonical_aggregates[select.aggregate].name, select.aggregate, "none")
    canonical.default = coalesce(select.default, canonical_aggregates[canonical.aggregate].default)

    if hasattr(unwrap(frum), "_normalize_select"):
        return frum._normalize_select(canonical)

    output = []
    if not select.value or select.value == ".":
        output.extend([
            set_default(
                {
                    "name": c.name,
                    "value": jx_expression(c.name, schema=schema)
                },
                canonical
            )
            for c in frum.get_leaves()
        ])
    elif is_text(select.value):
        if select.value.endswith(".*"):
            canonical.name = coalesce(select.name, ".")
            value = jx_expression(select[:-2], schema=schema)
            if not is_op(value, Variable):
                Log.error("`*` over general expression not supported yet")
                output.append([
                    set_default(
                        {
                            "value": LeavesOp(value, prefix=select.prefix),
                            "format": "dict"  # MARKUP FOR DECODING
                        },
                        canonical
                    )
                    for c in frum.get_columns()
                    if c.jx_type not in STRUCT
                ])
            else:
                Log.error("do not know what to do")
        else:
            canonical.name = coalesce(select.name, select.value, select.aggregate)
            canonical.value = jx_expression(select.value, schema=schema)
            output.append(canonical)

    output = wrap(output)
    if any(n==None for n in output.name):
        Log.error("expecting select to have a name: {{select}}", select=select)
    return output