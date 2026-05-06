def create_table(table, data):
    """
    Create table with defined name and fields
    :return: None
    """

    fields = data['fields']
    query = '('
    indexed_fields = ''

    for key, value in fields.items():
        non_case_field = value[0][0:value[0].find('(')]

        if non_case_field == 'int':
            sign = value[0][value[0].find(',') + 1:-1:].strip()
            if sign == 'signed':
                field_type = 'Int'
            else:
                field_type = 'UInt'

            bits = re.findall('\d+', value[0])[0]
            field = key + ' ' + field_type + bits
            query += field + ','

        elif non_case_field == 'strin':
            field_type = 'String'
            field = key + ' ' + field_type
            query += field + ','

        elif non_case_field == 'float':
            field_type = 'Float'
            bits = re.findall('\d+', value[0])[0]
            field = key + ' ' + field_type + bits
            query += field + ','

        if value[1] == 'yes':
            indexed_fields += key + ','

    query = query[:-1:] + f",date Date) ENGINE = MergeTree(date, ({indexed_fields} date), 8192)"

    client.execute(f"CREATE TABLE {table} {query}")