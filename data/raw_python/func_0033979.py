def readTableFromDelimited(f, separator="\t"):
    """
    Reads a table object from given plain delimited file.
    """
    rowNames = []
    columnNames = []
    matrix = []

    first = True
    for line in f.readlines():
        line = line.rstrip()
        if len(line) == 0:
            continue

        row = line.split(separator)
        if first:
            columnNames = row[1:]
            first = False
        else:
            rowNames.append(row[0])
            matrix.append([float(c) for c in row[1:]])

    return Table(rowNames, columnNames, matrix)