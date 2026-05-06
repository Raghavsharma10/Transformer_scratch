def sheetDelete(book=None,sheet=None):
    """
    Delete a sheet from a book. If either isn't given, use the active one.
    """
    if book is None:
        book=activeBook()
    if sheet in sheetNames():
        PyOrigin.WorksheetPages(book).Layers(sheetNames().index(sheet)).Destroy()