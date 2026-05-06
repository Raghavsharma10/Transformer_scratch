def create_index(column_names, unique=False):
    """
    Create a new index of the columns in column_names, where column_names is
    a list of strings. If unique is True, it will be a
    unique index.
    """
    connection = _State.connection()
    _State.reflect_metadata()
    table_name = _State.table.name

    table = _State.table

    index_name = re.sub(r'[^a-zA-Z0-9]', '', table_name) + '_'
    index_name += '_'.join(re.sub(r'[^a-zA-Z0-9]', '', x)
                           for x in column_names)

    if unique:
        index_name += '_unique'

    columns = []
    for column_name in column_names:
        columns.append(table.columns[column_name])

    current_indices = [x.name for x in table.indexes]
    index = sqlalchemy.schema.Index(index_name, *columns, unique=unique)
    if index.name not in current_indices:
        index.create(bind=_State.engine)