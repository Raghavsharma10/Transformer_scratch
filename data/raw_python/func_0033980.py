def readTableFromCSV(f, dialect="excel"):
    """
    Reads a table object from given CSV file.
    """
    rowNames = []
    columnNames = []
    matrix = []

    first = True
    for row in csv.reader(f, dialect):
        if first:
            columnNames = row[1:]
            first = False
        else:
            rowNames.append(row[0])
            matrix.append([float(c) for c in row[1:]])

    return Table(rowNames, columnNames, matrix)