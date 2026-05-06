def _normalize_select_no_context(select, schema=None):
    """
    SAME NORMALIZE, BUT NO SOURCE OF COLUMNS
    """
    if not _Column:
        _late_import()

    if is_text(select):
        select = Data(value=select)
    else:
        select = wrap(select)

    output = select.copy()
    if not select.value:
        output.name = coalesce(select.name, select.aggregate)
        if output.name:
            output.value = jx_expression(".", schema=schema)
        else:
            return Null
    elif is_text(select.value):
        if select.value.endswith(".*"):
            name = select.value[:-2].lstrip(".")
            output.name = coalesce(select.name,  name)
            output.value = LeavesOp(Variable(name), prefix=coalesce(select.prefix, name))
        else:
            if select.value == ".":
                output.name = coalesce(select.name, select.aggregate, ".")
                output.value = jx_expression(select.value, schema=schema)
            elif select.value == "*":
                output.name = coalesce(select.name, select.aggregate, ".")
                output.value = LeavesOp(Variable("."))
            else:
                output.name = coalesce(select.name, select.value.lstrip("."), select.aggregate)
                output.value = jx_expression(select.value, schema=schema)
    elif is_number(output.value):
        if not output.name:
            output.name = text_type(output.value)
        output.value = jx_expression(select.value, schema=schema)
    else:
        output.value = jx_expression(select.value, schema=schema)

    if not output.name:
        Log.error("expecting select to have a name: {{select}}",  select= select)
    if output.name.endswith(".*"):
        Log.error("{{name|quote}} is invalid select", name=output.name)

    output.aggregate = coalesce(canonical_aggregates[select.aggregate].name, select.aggregate, "none")
    output.default = coalesce(select.default, canonical_aggregates[output.aggregate].default)
    return output