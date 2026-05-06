def get_filter(q: tldap.Q, fields: Dict[str, tldap.fields.Field], pk: str):
    """
    Translate the Q tree into a filter string to search for, or None
    if no results possible.
    """
    # check the details are valid
    if q.negated and len(q.children) == 1:
        op = b"!"
    elif q.connector == tldap.Q.AND:
        op = b"&"
    elif q.connector == tldap.Q.OR:
        op = b"|"
    else:
        raise ValueError("Invalid value of op found")

    # scan through every child
    search = []
    for child in q.children:
        # if this child is a node, then descend into it
        if isinstance(child, tldap.Q):
            search.append(get_filter(child, fields, pk))
        else:
            # otherwise get the values in this node
            name, value = child

            # split the name if possible
            name, _, operation = name.rpartition("__")
            if name == "":
                name, operation = operation, None

            # replace pk with the real attribute
            if name == "pk":
                name = pk

            # DN is a special case
            if name == "dn":
                dn_name = "entryDN:"
                if isinstance(value, list):
                    s = []
                    for v in value:
                        assert isinstance(v, str)
                        v = v.encode('utf_8')
                        s.append(get_filter_item(dn_name, operation, v))
                    search.append("(&".join(search) + ")")

                # or process just the single value
                else:
                    assert isinstance(value, str)
                    v = value.encode('utf_8')
                    search.append(get_filter_item(dn_name, operation, v))
                continue

            # try to find field associated with name
            field = fields[name]
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
                assert isinstance(value, str)

            # process as list
            if isinstance(value, list):
                s = []
                for v in value:
                    v = field.value_to_filter(v)
                    s.append(get_filter_item(name, operation, v))
                search.append(b"(&".join(search) + b")")

            # or process just the single value
            else:
                value = field.value_to_filter(value)
                search.append(get_filter_item(name, operation, value))

    # output the results
    if len(search) == 1 and not q.negated:
        # just one non-negative term, return it
        return search[0]
    else:
        # multiple terms
        return b"(" + op + b"".join(search) + b")"