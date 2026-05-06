def titletable(html_doc, tofloat=True):
    """return a list of [(title, table), .....]

    title = previous item with a <b> tag
    table = rows -> [[cell1, cell2, ..], [cell1, cell2, ..], ..]"""
    soup = BeautifulSoup(html_doc, "html.parser")
    btables = soup.find_all(['b', 'table']) # find all the <b> and <table>
    titletables = []
    for i, item in enumerate(btables):
        if item.name == 'table':
            for j in range(i + 1):
                if btables[i-j].name == 'b':# step back to find a <b>
                    break
            titletables.append((btables[i - j], item))
    if tofloat:
        t2m = table2val_matrix
    else:
        t2m = table2matrix
    titlerows = [(tl.contents[0], t2m(tb)) for tl, tb in titletables]
    return titlerows