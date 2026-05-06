def normalize_query_parameters(query_string):
    """
    normalize_query_parameters(query_string) -> dict

    Converts a query string into a dictionary mapping parameter names to a
    list of the sorted values.  This ensurses that the query string follows
    % encoding rules according to RFC 3986 and checks for duplicate keys.

    A ValueError exception is raised if a percent encoding is invalid.
    """
    if query_string == "":
        return {}

    components = query_string.split("&")
    result = {}

    for component in components:
        try:
            key, value = component.split("=", 1)
        except ValueError:
            key = component
            value = ""

        if component == "":
            # Empty component; skip it.
            continue
        
        key = normalize_uri_path_component(key)
        value = normalize_uri_path_component(value)

        if key in result:
            result[key].append(value)
        else:
            result[key] = [value]

    return dict([(key, sorted(values))
                 for key, values in iteritems(result)])