def getPageType(name,number=False):
    """Returns the type of the page with that name.
    If that name doesn't exist, None is returned.

    Args:
        name (str): name of the page to get the folder from
        number (bool): if True, return numbers (i.e., a graph will be 3)
            if False, return words where appropriate (i.e, "graph")

    Returns:
        string of the type of object the page is
    """
    if not name in pageNames():
        return None
    pageType=PyOrigin.Pages(name).GetType()
    if number:
        return str(pageType)
    if pageType==1:
        return "matrix"
    if pageType==2:
        return "book"
    if pageType==3:
        return "graph"
    if pageType==4:
        return "layout"
    if pageType==5:
        return "notes"