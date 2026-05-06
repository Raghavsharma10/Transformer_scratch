def rows_to_columns(data, schema=None):
    """
    :param data: array of objects
    :param schema: Known schema, will be extended to include all properties found in data
    :return: Table
    """
    if not schema:
        schema = SchemaTree()
    all_schema = schema
    all_leaves = schema.leaves
    values = {full_name: [] for full_name in all_leaves}
    reps = {full_name: [] for full_name in all_leaves}
    defs = {full_name: [] for full_name in all_leaves}

    def _none_to_column(schema, path, rep_level, def_level):
        for full_path in all_schema.leaves:
            if startswith_field(full_path, path):
                reps[full_path].append(rep_level)
                defs[full_path].append(def_level)

    def _value_to_column(value, schema, path, counters, def_level):
        ptype = type(value)
        ntype, dtype, ltype, jtype, itype, byte_width = python_type_to_all_types[ptype]

        if jtype is NESTED:
            if schema.element.repetition_type != REPEATED:
                Log.error("Expecting {{path|quote}} to be repeated", path=path)

            new_path = path
            if not value:
                _none_to_column(schema, new_path, get_rep_level(counters), def_level)
            else:
                try:
                    new_schema = schema.more.get('.')

                    if not new_schema:
                        if schema.locked:
                            # DEFAULT TO REQUIRED ENTRIES
                            new_schema = schema
                            schema.element.repetition_type = REQUIRED
                        else:
                            new_path = path
                            new_value = value[0]
                            ptype = type(new_value)
                            new_schema = schema.add(
                                new_path,
                                OPTIONAL,
                                ptype
                            )
                            if new_value is None or python_type_to_json_type[ptype] in PRIMITIVE:
                                values[new_path] = []
                                reps[new_path] = [0] * counters[0]
                                defs[new_path] = [0] * counters[0]
                    for k, new_value in enumerate(value):
                        new_counters = counters + (k,)
                        _value_to_column(new_value, new_schema, new_path, new_counters, def_level+1)
                finally:
                    schema.element.repetition_type = REPEATED
        elif jtype is OBJECT:
            if value is None:
                if schema.element.repetition_type == REQUIRED:
                    Log.error("{{path|quote}} is required", path=path)
                _none_to_column(schema, path, get_rep_level(counters), def_level)
            else:
                if schema.element.repetition_type == REPEATED:
                    Log.error("Expecting {{path|quote}} to be repeated", path=path)

                if schema.element.repetition_type == REQUIRED:
                    new_def_level = def_level
                else:
                    counters = counters + (0,)
                    new_def_level = def_level+1

                for name, sub_schema in schema.more.items():
                    new_path = concat_field(path, name)
                    new_value = value.get(name, None)
                    _value_to_column(new_value, sub_schema, new_path, counters, new_def_level)

                for name in set(value.keys()) - set(schema.more.keys()):
                    if schema.locked:
                        Log.error("{{path}} is not allowed in the schema", path=path)
                    new_path = concat_field(path, name)
                    new_value = value.get(name, None)
                    ptype = type(new_value)
                    sub_schema = schema.add(
                        new_path,
                        REPEATED if isinstance(new_value, list) else OPTIONAL,
                        ptype
                    )
                    if python_type_to_json_type[ptype] in PRIMITIVE:
                        values[new_path] = []
                        reps[new_path] = [0] * counters[0]
                        defs[new_path] = [0] * counters[0]
                    _value_to_column(new_value, sub_schema, new_path, counters, new_def_level)
        else:
            if jtype is STRING:
                value = value.encode('utf8')
            merge_schema(schema, path, value)
            values[path].append(value)
            if schema.element.repetition_type == REQUIRED:
                reps[path].append(get_rep_level(counters))
                defs[path].append(def_level)
            else:
                reps[path].append(get_rep_level(counters))
                defs[path].append(def_level + 1)

    for rownum, new_value in enumerate(data):
        try:
            _value_to_column(new_value, schema, '.', (rownum,), 0)
        except Exception as e:
            Log.error("can not encode {{row|json}}", row=new_value, cause=e)

    return Table(values, reps, defs, len(data), schema)