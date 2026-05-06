def mkrows(l, pad, width, height):
    '''
    Compute the optimal number of rows based on our lists' largest element and
    our terminal size in columns and rows.

    Work out our maximum column number by dividing the width of the terminal by
    our largest element.

    While the length of our list is greater than the total number of elements we
    can fit on the screen increment the height by one.
    '''
    maxcols = int(width/pad)
    while len(l) > height * maxcols:
        height += 1
    return height