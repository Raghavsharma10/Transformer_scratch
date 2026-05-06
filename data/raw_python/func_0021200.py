def schema_to_index(schema, index_names=None):
    """Get index/doc_type given a schema URL.

    :param schema: The schema name
    :param index_names: A list of index name.
    :returns: A tuple containing (index, doc_type).
    """
    parts = schema.split('/')
    doc_type = os.path.splitext(parts[-1])

    if doc_type[1] not in {'.json', }:
        return (None, None)

    if index_names is None:
        return (build_index_name(current_app, *parts), doc_type[0])

    for start in range(len(parts)):
        index_name = build_index_name(current_app, *parts[start:])
        if index_name in index_names:
            return (index_name, doc_type[0])

    return (None, None)