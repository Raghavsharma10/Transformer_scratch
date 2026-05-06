def refine(query, description):
    """
    Refine a query from a dictionary of parameters that describes it.
    See `describe` for more information.
    """

    for attribute, arguments in description.items():
        if hasattr(query, attribute):
            attribute = getattr(query, attribute)
        else:
            raise ValueError("Unknown query method: " + attribute)

        # query descriptions are often automatically generated, and
        # may include empty calls, which we skip
        if utils.isempty(arguments):
            continue

        if callable(attribute):
            method = attribute
            if isinstance(arguments, dict):
                query = method(**arguments)
            elif isinstance(arguments, list):
                query = method(*arguments)
            else:
                query = method(arguments)
        else:
            setattr(attribute, arguments)

    return query