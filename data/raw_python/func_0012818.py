def lines_table(html_doc, tofloat=True):
    """return a list of [(lines, table), .....]

    lines = all the significant lines before the table.
        These are lines between this table and
        the previous table or 'hr' tag
    table = rows -> [[cell1, cell2, ..], [cell1, cell2, ..], ..]

    The lines act as a description for what is in the table
    """
    soup = BeautifulSoup(html_doc, "html.parser")
    linestables = []
    elements = soup.p.next_elements # start after the first para
    for element in elements:
        tabletup = []
        if not _has_name(element):
            continue
        if element.name == 'table': # hit the first table
            beforetable = []
            prev_elements = element.previous_elements # walk back and get the lines
            for prev_element in prev_elements:
                if not _has_name(prev_element):
                    continue
                if prev_element.name not in ('br', None): # no lines here
                    if prev_element.name in ('table', 'hr', 'tr', 'td'):
                        # just hit the previous table. You got all the lines
                        break
                    if prev_element.parent.name == "p":
                        # if the parent is "p", you will get it's text anyways from the parent
                        pass
                    else:
                        if prev_element.get_text(): # skip blank lines
                            beforetable.append(prev_element.get_text())
            beforetable.reverse()
            tabletup.append(beforetable)
            function_selector = {True:table2val_matrix, False:table2matrix}
            function = function_selector[tofloat]
            tabletup.append(function(element))
        if tabletup:
            linestables.append(tabletup)
    return linestables