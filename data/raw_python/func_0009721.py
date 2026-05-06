def save_var(name, value):
    """
    Save a variable to the table specified by _State.vars_table_name. Key is
    the name of the variable, and value is the value.
    """
    connection = _State.connection()
    _State.reflect_metadata()

    vars_table = sqlalchemy.Table(
        _State.vars_table_name, _State.metadata,
        sqlalchemy.Column('name', sqlalchemy.types.Text, primary_key=True),
        sqlalchemy.Column('value_blob', sqlalchemy.types.LargeBinary),
        sqlalchemy.Column('type', sqlalchemy.types.Text),
        keep_existing=True
    )

    vars_table.create(bind=connection, checkfirst=True)

    column_type = get_column_type(value)

    if column_type == sqlalchemy.types.LargeBinary:
        value_blob = value
    else:
        value_blob = unicode(value).encode('utf-8')

    values = dict(name=name,
                  value_blob=value_blob,
                  # value_blob=Blob(value),
                  type=column_type.__visit_name__.lower())

    vars_table.insert(prefixes=['OR REPLACE']).values(**values).execute()