def select_from_table(fields, table, query):
    """
    SQL query for selecting data from certain table
    :return: selected data
    """

    select_data = client.execute(f"SELECT {fields} FROM {table} {query}", with_column_types=True)

    keys = [i[0] for i in select_data[1]]

    result = []

    for i in range(len(select_data[0])):
        tmp = []

        for j in range(len(keys)):
            tmp.append(str(select_data[0][i][j]))

        result.append(dict(list(zip(keys, tmp))))

    # for i in range(len(select_data[0])):
    #     result.append(dict(list(zip(keys, select_data[0][i]))))

    return result