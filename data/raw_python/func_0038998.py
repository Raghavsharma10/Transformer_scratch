def extract_attrs(x, n):
    """Extracts attributes from element list 'x' beginning at index 'n'.

    The elements encapsulating the attributes (typically a series of Str and
    Space elements) are removed from 'x'.  Items before index 'n' are left
    unchanged.

    Returns the attributes in pandoc format.  A ValueError is raised if
    attributes aren't found.  An IndexError is raised if the index 'n' is out
    of range."""

    # Check for the start of the attributes string
    if not (x[n]['t'] == 'Str' and x[n]['c'].startswith('{')):
        raise ValueError('Attributes not found.')

    # It starts with {, so this *may* be an attributes list.  Search for where
    # the attributes end.  Do not consider } in quoted elements.

    seq = []          # A sequence of saved values
    quotechar = None  # Used to keep track of quotes in strings
    flag = False      # Flags that an attributes list was found
    i = 0             # Initialization

    for i, v in enumerate(x[n:]):  # Scan through the list
        if v and v['t'] == 'Str':
            # Scan for } outside of a quote
            for j, c in enumerate(v['c']):
                if c == quotechar:  # This is an end quote
                    quotechar = None
                elif c in ['"', "'"]:  # This is an open quote
                    quotechar = c
                elif c == '}' and quotechar is None:  # The attributes end here
                    # Split the string at the } and save the pieces
                    head, tail = v['c'][:j+1], v['c'][j+1:]
                    x[n+i] = copy.deepcopy(v)
                    x[n+i]['c'] = tail
                    v['c'] = head
                    flag = True
                    break
        seq.append(v)
        if flag:
            break

    if flag:  # Attributes string was found, so process it

        # Delete empty and extracted elements
        if x[n+i]['t'] == 'Str' and not x[n+i]['c']:
            del x[n+i]
        del x[n:n+i]

        # Process the attrs
        attrstr = stringify(dollarfy(quotify(seq))).strip()
        attrs = PandocAttributes(attrstr, 'markdown').to_pandoc()

        # Remove extranneous quotes from kvs
        for i, (k, v) in enumerate(attrs[2]):  # pylint: disable=unused-variable
            if v[0] == v[-1] == '"' or v[0] == "'" == v[-1] == "'":
                attrs[2][i][1] = attrs[2][i][1][1:-1]

        # We're done
        return attrs

    # Attributes not found
    raise ValueError('Attributes not found.')