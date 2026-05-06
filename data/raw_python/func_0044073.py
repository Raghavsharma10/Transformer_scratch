async def drop(**data):
    """
    RPC method for deleting table from the database
    :return None
    """

    table = data['table']

    try:
        clickhouse_queries.drop_table(table)
        return 'Table was successfully deleted'

    except ServerException as e:
        exception_code = int(str(e)[5:8].strip())
        if exception_code == 60:
            return 'Table does not exists'