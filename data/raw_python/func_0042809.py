def parse_properties(parent_index_name, parent_name, nested_path, esProperties):
    """
    RETURN THE COLUMN DEFINITIONS IN THE GIVEN esProperties OBJECT
    """
    columns = FlatList()
    for name, property in esProperties.items():
        index_name = parent_index_name
        column_name = concat_field(parent_name, name)
        jx_name = column_name

        if property.type == "nested" and property.properties:
            # NESTED TYPE IS A NEW TYPE DEFINITION
            # MARKUP CHILD COLUMNS WITH THE EXTRA DEPTH
            self_columns = parse_properties(index_name, column_name, [column_name] + nested_path, property.properties)
            columns.extend(self_columns)
            columns.append(Column(
                name=jx_name,
                es_index=index_name,
                es_column=column_name,
                es_type="nested",
                jx_type=NESTED,
                last_updated=Date.now(),
                nested_path=nested_path
            ))

            continue

        if property.properties:
            child_columns = parse_properties(index_name, column_name, nested_path, property.properties)
            columns.extend(child_columns)
            columns.append(Column(
                name=jx_name,
                es_index=index_name,
                es_column=column_name,
                es_type="source" if property.enabled == False else "object",
                jx_type=OBJECT,
                last_updated=Date.now(),
                nested_path=nested_path
            ))

        if property.dynamic:
            continue
        if not property.type:
            continue

        cardinality = 0 if not (property.store or property.enabled) and name != '_id' else None

        if property.fields:
            child_columns = parse_properties(index_name, column_name, nested_path, property.fields)
            if cardinality is None:
                for cc in child_columns:
                    cc.cardinality = None
            columns.extend(child_columns)

        if property.type in es_type_to_json_type.keys():
            columns.append(Column(
                name=jx_name,
                es_index=index_name,
                es_column=column_name,
                es_type=property.type,
                jx_type=es_type_to_json_type[property.type],
                cardinality=cardinality,
                last_updated=Date.now(),
                nested_path=nested_path
            ))
            if property.index_name and name != property.index_name:
                columns.append(Column(
                    name=jx_name,
                    es_index=index_name,
                    es_column=column_name,
                    es_type=property.type,
                    jx_type=es_type_to_json_type[property.type],
                    cardinality=0 if property.store else None,
                    last_updated=Date.now(),
                    nested_path=nested_path
                ))
        elif property.enabled == None or property.enabled == False:
            columns.append(Column(
                name=jx_name,
                es_index=index_name,
                es_column=column_name,
                es_type="source" if property.enabled == False else "object",
                jx_type=OBJECT,
                cardinality=0 if property.store else None,
                last_updated=Date.now(),
                nested_path=nested_path
            ))
        else:
            Log.warning("unknown type {{type}} for property {{path}}", type=property.type, path=parent_name)

    return columns