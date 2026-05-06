def _select1(data, field, depth, output):
    """
    SELECT A SINGLE FIELD
    """
    for d in data:
        for i, f in enumerate(field[depth:]):
            d = d[f]
            if d == None:
                output.append(None)
                break
            elif is_list(d):
                _select1(d, field, i + 1, output)
                break
        else:
            output.append(d)