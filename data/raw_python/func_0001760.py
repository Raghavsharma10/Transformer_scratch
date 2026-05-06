def extract_table(tabletag):
    # type: (Tag) -> List[Dict]
    """
    Extract HTML table as list of dictionaries

    Args:
        tabletag (Tag): BeautifulSoup tag

    Returns:
        str: Text of tag stripped of leading and trailing whitespace and newlines and with &nbsp replaced with space

    """
    theadtag = tabletag.find_next('thead')

    headertags = theadtag.find_all('th')
    if len(headertags) == 0:
        headertags = theadtag.find_all('td')
    headers = []
    for tag in headertags:
        headers.append(get_text(tag))

    tbodytag = tabletag.find_next('tbody')
    trtags = tbodytag.find_all('tr')

    table = list()
    for trtag in trtags:
        row = dict()
        tdtags = trtag.find_all('td')
        for i, tag in enumerate(tdtags):
            row[headers[i]] = get_text(tag)
        table.append(row)
    return table