def _normalize_sort(sort=None):
    """
    CONVERT SORT PARAMETERS TO A NORMAL FORM SO EASIER TO USE
    """

    if sort==None:
        return FlatList.EMPTY

    output = FlatList()
    for s in listwrap(sort):
        if is_text(s):
            output.append({"value": jx_expression(s), "sort": 1})
        elif is_expression(s):
            output.append({"value": s, "sort": 1})
        elif mo_math.is_integer(s):
            output.append({"value": jx_expression({"offset": s}), "sort": 1})
        elif not s.sort and not s.value and all(d in sort_direction for d in s.values()):
            for v, d in s.items():
                output.append({"value": jx_expression(v), "sort": sort_direction[d]})
        elif not s.sort and not s.value:
            Log.error("`sort` clause must have a `value` property")
        else:
            output.append({"value": jx_expression(coalesce(s.value, s.field)), "sort": sort_direction[s.sort]})
    return output