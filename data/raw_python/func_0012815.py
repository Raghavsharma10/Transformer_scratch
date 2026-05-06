def table2val_matrix(table):
    """convert a table to a list of lists - a 2D matrix
    Converts numbers to float"""
    if not is_simpletable(table):
        raise NotSimpleTable("Not able read a cell in the table as a string")
    rows = []
    for tr in table('tr'):
        row = []
        for td in tr('td'):
            td = tdbr2EOL(td)
            try:
                val = td.contents[0]
            except IndexError:
                row.append('')
            else:
                try:
                    val = float(val)
                    row.append(val)
                except ValueError:
                    row.append(val)
        rows.append(row)
    return rows