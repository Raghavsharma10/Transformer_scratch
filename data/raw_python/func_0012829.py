def row2table(soup, table, row):
    """ad a row to the table"""
    tr = Tag(soup, name="tr")
    table.append(tr)
    for attr in row:
        td = Tag(soup, name="td")
        tr.append(td)
        td.append(attr)