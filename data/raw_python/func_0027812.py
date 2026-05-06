def pageNames(matching=False,workbooks=True,graphs=True):
    """
    Returns the names of everything (books, notes, graphs, etc.) in the project.

    Args:
        matching (str, optional): if given, only return names with this string in it
        workbooks (bool): if True, return workbooks
        graphs (bool): if True, return workbooks

    Returns:
        A list of the names of what you requested
    """
    # first collect the pages we want
    pages=[]
    if workbooks:
        pages.extend(PyOrigin.WorksheetPages())
    if graphs:
        pages.extend(PyOrigin.GraphPages())

    # then turn them into a list of strings
    pages = [x.GetName() for x in pages]

    # do our string matching if it's needed
    if matching:
        pages=[x for x in pages if matching in x]
    return pages