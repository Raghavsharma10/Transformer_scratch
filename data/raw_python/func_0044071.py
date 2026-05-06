async def insert(**data):
    """
    RPC method for inserting data to the table
    :return: None
    """

    table = data.get('table')

    try:
        clickhouse_queries.insert_into_table(table, data)
        return 'Data was successfully inserted into table'

    except ServerException as e:
        exception_code = int(str(e)[5:8].strip())

        if exception_code == 60:
            return 'Table does not exists'
        elif exception_code == 50:
            return 'Invalid params'