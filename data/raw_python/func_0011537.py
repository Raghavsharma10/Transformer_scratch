def FSeek(params, ctxt, scope, stream, coord):
    """Returns 0 if successful or -1 if the address is out of range
    """
    if len(params) != 1:
        raise errors.InvalidArguments(coord, "{} args".format(len(params)), "FSeek accepts only one argument")
    
    pos = PYVAL(params[0])
    curr_pos = stream.tell()

    fsize = stream.size()

    if pos > fsize:
        stream.seek(fsize)
        return -1
    elif pos < 0:
        stream.seek(0)
        return -1

    diff = pos - curr_pos
    if diff < 0:
        stream.seek(pos)
        return 0

    data = stream.read(diff)

    # let the ctxt automatically append numbers, as needed, unless the previous
    # child was also a skipped field
    skipped_name = "_skipped"

    if len(ctxt._pfp__children) > 0 and ctxt._pfp__children[-1]._pfp__name.startswith("_skipped"):
        old_name = ctxt._pfp__children[-1]._pfp__name
        data = ctxt._pfp__children[-1].raw_data + data
        skipped_name = old_name
        ctxt._pfp__children = ctxt._pfp__children[:-1]
        del ctxt._pfp__children_map[old_name]

    tmp_stream = bitwrap.BitwrappedStream(six.BytesIO(data))
    new_field = pfp.fields.Array(len(data), pfp.fields.Char, tmp_stream)
    ctxt._pfp__add_child(skipped_name, new_field, stream)
    scope.add_var(skipped_name, new_field)

    return 0