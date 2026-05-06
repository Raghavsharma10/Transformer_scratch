def is_simpletable(table):
    """test if the table has only strings in the cells"""
    tds = table('td')
    for td in tds:
        if td.contents != []:
            td = tdbr2EOL(td)
            if len(td.contents) == 1:
                thecontents = td.contents[0]
                if not isinstance(thecontents, NavigableString):
                    return False
            else:
                return False
    return True