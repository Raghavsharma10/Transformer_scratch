def _get_members(obj, traverse_bases=True, filter=default_filter,
                 recursive=True):
    """Retrieve the member attributes of a module or a class.

    The descriptor protocol is bypassed."""
    if filter is None:
        filter = _true

    out = []
    stack = collections.deque((obj,))
    while stack:
        obj = stack.popleft()
        if traverse_bases and isinstance(obj, _CLASS_TYPES):
            roots = [base for base in inspect.getmro(obj)
                     if base not in (type, object)]
        else:
            roots = [obj]

        members = []
        seen = set()
        for root in roots:
            for name, value in _iteritems(getattr(root, '__dict__', {})):
                if name not in seen and filter(name, value):
                    members.append((name, value))

                seen.add(name)

        members = sorted(members)
        for _, value in members:
            if recursive and isinstance(value, _CLASS_TYPES):
                stack.append(value)

        out.extend(members)

    return out