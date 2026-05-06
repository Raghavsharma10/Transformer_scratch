async def select(**data):
    """
    RPC method for selecting data from the database
    :return selected data
    """

    try:
        select_data = clickhouse_queries.select_from_table(table=data['table'], query=data['query'], fields=data['fields'])
        return str(select_data)

    except ServerException as e:
        exception_code = int(str(e)[5:8].strip())

        if exception_code == 60:
            return 'Table does not exists'
        elif exception_code == 50:
            return 'Invalid params'