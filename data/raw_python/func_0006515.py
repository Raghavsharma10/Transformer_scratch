def mkcols(l, rows):
    '''
    Compute the size of our columns by first making them a divisible of our row
    height and then splitting our list into smaller lists the size of the row
    height.
    '''
    cols = []
    base = 0
    while len(l) > rows and len(l) % rows != 0:
        l.append("")
    for i in range(rows, len(l) + rows, rows):
        cols.append(l[base:i])
        base = i
    return cols