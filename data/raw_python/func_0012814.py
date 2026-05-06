def table2matrix(table):
    """convert a table to a list of lists - a 2D matrix"""

    if not is_simpletable(table):
        raise NotSimpleTable("Not able read a cell in the table as a string")
    rows = []
    for tr in table('tr'):
        row = []
        for td in tr('td'):
            td = tdbr2EOL(td) # convert any '<br>' in the td to line ending
            try:
                row.append(td.contents[0])
            except IndexError:
                row.append('')
        rows.append(row)
    return rows