def fit_row(connection, row, unique_keys):
    """
    Takes a row and checks to make sure it fits in the columns of the
    current table. If it does not fit, adds the required columns.
    """
    new_columns = []
    for column_name, column_value in list(row.items()):
        new_column = sqlalchemy.Column(column_name,
                                       get_column_type(column_value))

        if not column_name in list(_State.table.columns.keys()):
            new_columns.append(new_column)
            _State.table.append_column(new_column)

    if _State.table_pending:
        create_table(unique_keys)
        return

    for new_column in new_columns:
        add_column(connection, new_column)