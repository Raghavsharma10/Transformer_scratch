def getcols(sheetMatch=None,colMatch="Decay"):
    """find every column in every sheet and put it in a new sheet or book."""
    book=BOOK()
    if sheetMatch is None:
        matchingSheets=book.sheetNames
        print('all %d sheets selected '%(len(matchingSheets)))
    else:
        matchingSheets=[x for x in book.sheetNames if sheetMatch in x]
        print('%d of %d sheets selected matching "%s"'%(len(matchingSheets),len(book.sheetNames),sheetMatch))
    matchingSheetsWithCol=[]
    for sheetName in matchingSheets:
        i = book.sheetNames.index(sheetName) # index of that sheet
        for j,colName in enumerate(book.sheets[i].colDesc):
            if colMatch in colName:
                matchingSheetsWithCol.append((sheetName,j))
                break
        else:
            print("  no match in [%s]%s"%(book.bookName,sheetName))
    print("%d of %d of those have your column"%(len(matchingSheetsWithCol),len(matchingSheets)))
    for item in matchingSheetsWithCol:
        print(item,item[0],item[1])