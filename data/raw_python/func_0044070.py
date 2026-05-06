async def create_table(**data):
    """
    RPC method for creating table with custom name and fields
    :return event id
    """

    table = data.get('table')

    try:
        clickhouse_queries.create_table(table, data)
        return 'Table was successfully created'

    except ServerException as e:
        exception_code = int(str(e)[5:8].strip())

        if exception_code == 57:
            return 'Table already exists'
        elif exception_code == 50:
            return 'Invalid params'