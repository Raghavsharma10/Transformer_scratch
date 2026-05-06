def getSheet(book=None,sheet=None):
    """returns the pyorigin object for a sheet."""

    # figure out what book to use
    if book and not book.lower() in [x.lower() for x in bookNames()]:
        print("book %s doesn't exist"%book)
        return
    if book is None:
        book=activeBook().lower()
    if book is None:
        print("no book given or selected")
        return

    # figure out what sheet to use
    if sheet and not sheet.lower() in [x.lower() for x in sheetNames(book)]:
        print("sheet %s doesn't exist"%sheet)
        return
    if sheet is None:
        sheet=activeSheet().lower()
    if sheet is None:
        return("no sheet given or selected")
        print

    # by now, we know the book/sheet exists and can be found
    for poSheet in PyOrigin.WorksheetPages(book).Layers():
        if poSheet.GetName().lower()==sheet.lower():
            return poSheet
    return False