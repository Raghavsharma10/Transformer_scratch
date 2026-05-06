def aggs_iterator(aggs, es_query, decoders, give_me_zeros=False):
    """
    DIG INTO ES'S RECURSIVE aggs DATA-STRUCTURE:
    RETURN AN ITERATOR OVER THE EFFECTIVE ROWS OF THE RESULTS

    :param aggs: ES AGGREGATE OBJECT
    :param es_query: THE ABSTRACT ES QUERY WE WILL TRACK ALONGSIDE aggs
    :param decoders: TO CONVERT PARTS INTO COORDINATES
    """
    coord = [0] * len(decoders)
    parts = deque()
    stack = []

    gen = _children(aggs, es_query.children)
    while True:
        try:
            index, c_agg, c_query, part = gen.next()
        except StopIteration:
            try:
                gen = stack.pop()
            except IndexError:
                return
            parts.popleft()
            continue

        if c_agg.get('doc_count') == 0 and not give_me_zeros:
            continue
        parts.appendleft(part)
        for d in c_query.decoders:
            coord[d.edge.dim] = d.get_index(tuple(p for p in parts if p is not None), c_query, index)

        children = c_query.children
        selects = c_query.selects
        if selects or not children:
            parts.popleft()  # c_agg WAS ON TOP
            yield (
                tuple(p for p in parts if p is not None),
                tuple(coord),
                c_agg,
                selects
            )
            continue

        stack.append(gen)
        gen = _children(c_agg, children)