def unpack_glist(g, type_, transfer_full=True):
    """Takes a glist, copies the values casted to type_ in to a list
    and frees all items and the list.
    """

    values = []
    item = g
    while item:
        ptr = item.contents.data
        value = cast(ptr, type_).value
        values.append(value)
        if transfer_full:
            free(ptr)
        item = item.next()
    if transfer_full:
        g.free()
    return values