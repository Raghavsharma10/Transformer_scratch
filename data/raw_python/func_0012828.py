def heading2table(soup, table, row):
    """add heading row to table"""
    tr = Tag(soup, name="tr")
    table.append(tr)
    for attr in row:
        th = Tag(soup, name="th")
        tr.append(th)
        th.append(attr)