def normalize_sort(sort=None):
    """
    CONVERT SORT PARAMETERS TO A NORMAL FORM SO EASIER TO USE
    """

    if not sort:
        return Null

    output = FlatList()
    for s in listwrap(sort):
        if is_text(s) or mo_math.is_integer(s):
            output.append({"value": s, "sort": 1})
        elif not s.field and not s.value and s.sort==None:
            #ASSUME {name: sort} FORM
            for n, v in s.items():
                output.append({"value": n, "sort": sort_direction[v]})
        else:
            output.append({"value": coalesce(s.field, s.value), "sort": coalesce(sort_direction[s.sort], 1)})
    return wrap(output)