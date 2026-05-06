def insert_into_table(table, data):
    """
    SQL query for inserting data into table
    :return: None
    """

    fields = data['fields']
    fields['date'] = datetime.datetime.now().date()

    query = '('

    for key in fields.keys():
        query += key + ','

    query = query[:-1:] + ")"

    client.execute(f"INSERT INTO {table} {query} VALUES", [tuple(fields.values())])