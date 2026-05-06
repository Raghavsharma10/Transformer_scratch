def _get_schema_from_list(frum, table_name, parent, nested_path, columns):
    """
    :param frum: The list
    :param table_name: Name of the table this list holds records for
    :param parent: parent path
    :param nested_path: each nested array, in reverse order
    :param columns: map from full name to column definition
    :return:
    """

    for d in frum:
        row_type = python_type_to_json_type[d.__class__]

        if row_type != "object":
            # EXPECTING PRIMITIVE VALUE
            full_name = parent
            column = columns[full_name]
            if not column:
                column = Column(
                    name=concat_field(table_name, full_name),
                    es_column=full_name,
                    es_index=".",
                    es_type=d.__class__.__name__,
                    jx_type=None,  # WILL BE SET BELOW
                    last_updated=Date.now(),
                    nested_path=nested_path,
                )
                columns.add(column)
            column.es_type = _merge_python_type(column.es_type, d.__class__)
            column.jx_type = python_type_to_json_type[column.es_type]
        else:
            for name, value in d.items():
                full_name = concat_field(parent, name)
                column = columns[full_name]
                if not column:
                    column = Column(
                        name=concat_field(table_name, full_name),
                        es_column=full_name,
                        es_index=".",
                        es_type=value.__class__.__name__,
                        jx_type=None,  # WILL BE SET BELOW
                        last_updated=Date.now(),
                        nested_path=nested_path,
                    )
                    columns.add(column)
                if is_container(value):  # GET TYPE OF MULTIVALUE
                    v = list(value)
                    if len(v) == 0:
                        this_type = none_type.__name__
                    elif len(v) == 1:
                        this_type = v[0].__class__.__name__
                    else:
                        this_type = reduce(
                            _merge_python_type, (vi.__class__.__name__ for vi in value)
                        )
                else:
                    this_type = value.__class__.__name__
                column.es_type = _merge_python_type(column.es_type, this_type)
                column.jx_type = python_type_to_json_type[column.es_type]

                if this_type in {"object", "dict", "Mapping", "Data"}:
                    _get_schema_from_list(
                        [value], table_name, full_name, nested_path, columns
                    )
                elif this_type in {"list", "FlatList"}:
                    np = listwrap(nested_path)
                    newpath = unwraplist([join_field(split_field(np[0]) + [name])] + np)
                    _get_schema_from_list(
                        value, table_name, full_name, newpath, columns
                    )