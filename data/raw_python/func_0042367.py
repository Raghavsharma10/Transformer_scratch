def values2rows(values, column_names):
    """
     CONVERT LIST OF JSON-IZABLE DATA STRUCTURE TO DATABASE ROW
     value - THE STRUCTURE TO CONVERT INTO row
     column_names - FOR ORDERING THE ALLOWED COLUMNS (EXTRA ATTRIBUTES ARE
                    LOST) THE COLUMN NAMES ARE EXPECTED TO HAVE dots (.)
                    FOR DEEPER PROPERTIES
    """
    values = wrap(values)
    lookup = {name: i for i, name in enumerate(column_names)}
    output = []
    for value in values:
        row = [None] * len(column_names)
        for k, v in value.leaves():
            index = lookup.get(k, -1)
            if index != -1:
                row[index] = v
        output.append(row)
    return output