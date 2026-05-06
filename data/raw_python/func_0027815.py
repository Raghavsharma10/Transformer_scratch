def sheetNames(book=None):
    """return sheet names of a book.

    Args:
        book (str, optional): If a book is given, pull names from
            that book. Otherwise, try the active one

    Returns:
        list of sheet names (typical case).
        None if book has no sheets.
        False if book doesn't exlist.

    """
    if book:
        if not book.lower() in [x.lower() for x in bookNames()]:
            return False
    else:
        book=activeBook()
    if not book:
        return False
    poBook=PyOrigin.WorksheetPages(book)
    if not len(poBook):
        return None
    return [x.GetName() for x in poBook.Layers()]