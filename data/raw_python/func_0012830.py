def printhtml(csvdiffs):
    """print the html"""
    soup = BeautifulSoup()
    html = Tag(soup, name="html")
    para1 = Tag(soup, name="p")
    para1.append(csvdiffs[0][0])
    para2 = Tag(soup, name="p")
    para2.append(csvdiffs[1][0])
    table = Tag(soup, name="table")
    table.attrs.update(dict(border="1"))

    soup.append(html)
    html.append(para1)
    html.append(para2)
    html.append(table)
    heading2table(soup, table, csvdiffs[3])
    for row in csvdiffs[4:]:
        row = [str(cell) for cell in row]
        row2table(soup, table, row)
    # print soup.prettify()
    print(soup)