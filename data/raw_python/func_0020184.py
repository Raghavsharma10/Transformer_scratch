def lookup(var_name, contexts=(), start=0):
    """lookup the value of the var_name on the stack of contexts

    :var_name: TODO
    :contexts: TODO
    :returns: None if not found

    """
    start = len(contexts) if start >=0 else start
    for context in reversed(contexts[:start]):
        try:
            if var_name in context:
                return context[var_name]
        except TypeError as te:
            # we may put variable on the context, skip it
            continue
    return None