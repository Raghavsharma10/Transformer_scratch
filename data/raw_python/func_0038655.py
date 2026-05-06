def insert_statement(table, columns, values):
    """Generate an insert statement string for dumping to text file or MySQL execution."""
    if not all(isinstance(r, (list, set, tuple)) for r in values):
        values = [[r] for r in values]
    rows = []
    for row in values:
        new_row = []
        for col in row:
            if col is None:
                new_col = 'NULL'
            elif isinstance(col, (int, float, Decimal)):
                new_col = str(MySQLConverterBase().to_mysql(col))
            else:
                string = str(MySQLConverterBase().to_mysql(col))
                if "'" in string:
                    new_col = '"' + string + '"'
                else:
                    new_col = "'" + string + "'"
            new_row.append(new_col)
        rows.append(', '.join(new_row))
    vals = '(' + '),\n\t('.join(rows) + ')'
    statement = "INSERT INTO\n\t{0} ({1}) \nVALUES\n\t{2}".format(wrap(table), cols_str(columns), vals)
    return statement