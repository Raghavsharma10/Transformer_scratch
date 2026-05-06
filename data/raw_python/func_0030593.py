def make_table_map(table, headers):
    """Create a function to map from rows with the structure of the headers to the structure of the table."""

    header_parts = {}
    for i, h in enumerate(headers):
        header_parts[h] = 'row[{}]'.format(i)

    body_code = 'lambda row: [{}]'.format(','.join(header_parts.get(c.name, 'None') for c in table.columns))
    header_code = 'lambda row: [{}]'.format(
        ','.join(header_parts.get(c.name, "'{}'".format(c.name)) for c in table.columns))

    return eval(header_code), eval(body_code)