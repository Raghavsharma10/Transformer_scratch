def sheetDeleteEmpty(bookName=None):
    """Delete all sheets which contain no data"""
    if bookName is None:
        bookName = activeBook()
    if not bookName.lower() in [x.lower() for x in bookNames()]:
        print("can't clean up a book that doesn't exist:",bookName)
        return
    poBook=PyOrigin.WorksheetPages(bookName)
    namesToKill=[]
    for i,poSheet in enumerate([poSheet for poSheet in poBook.Layers()]):
        poFirstCol=poSheet.Columns(0)
        if poFirstCol.GetLongName()=="" and poFirstCol.GetData()==[]:
            namesToKill.append(poSheet.GetName())
    for sheetName in namesToKill:
        print("deleting empty sheet",sheetName)
        sheetDelete(bookName,sheetName)