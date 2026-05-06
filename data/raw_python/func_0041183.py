def _add_element_by_names(src, names, value, override=False, digit=True):
    """
    Internal method recursive to Add element into a list or dict easily using
    a path.
    =============   =============   =======================================
    Parameter       Type            Description
    =============   =============   =======================================
    src             list or dict    element where add the value.
    names           list            list with names to navigate in src.
    value           ¿all?           value to add in src.
    override        boolean         Override the value in path src.
    =============   =============   =======================================
    Returns: src with added value
    """

    if src is None:
        return False

    else:

        if names and names[0]:
            head, *rest = names

            # list and digit head
            if isinstance(src, list):
                if force_list(digit)[0] and head.isdigit():
                    head = int(head)

                    # if src is a list and lenght <= head
                    if len(src) <= head:
                        src.extend([""] * (head + 1 - len(src)))

            # head not in src :(
            elif isinstance(src, dict):
                if head not in src:
                    src[head] = [""] * (int(rest[0]) + 1) if rest and force_list(digit)[0] and rest[0].isdigit() else {}

            # more heads in rest
            if rest:

                # Head find but isn't a dict or list to navigate for it.
                if not isinstance(src[head], (dict, list)):

                    # only could be str for dict or int for list
                    src[head] = [""] * (int(rest[0]) + 1) if force_list(digit)[0] and rest[0].isdigit() else {}

                    digit = digit if not digit or not isinstance(digit, list) else digit[1:]

                    if not force_list(digit)[0] and rest and str(rest[0]).isdigit() and isinstance(src[head], list) and override:
                        src[head] = {}

                    _add_element_by_names(src[head], rest, value, override=override, digit=digit)

                else:

                    digit = digit if not digit or not isinstance(digit, list) else digit[1:]

                    if not force_list(digit)[0] and rest and str(rest[0]).isdigit() and isinstance(src[head], list) and override:
                        src[head] = {}

                    _add_element_by_names(src[head], rest, value, override=override, digit=digit)

            # it's final head
            else:

                if not override:

                    if isinstance(src, list) and isinstance(head, int):

                        if src[head] == '':
                            src[head] = value
                        else:
                            src.append(value)

                    elif isinstance(src[head], list):
                        src[head].append(value)

                    elif isinstance(src[head], dict) and isinstance(value, dict):
                        src[head].update(value)

                    else:
                        src[head] = value

                else:
                    src[head] = value

        return src