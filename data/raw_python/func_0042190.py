def get_aligned_adjacent_coords(x, y):
    '''
    returns the nine clockwise adjacent coordinates on a keypad, where each row is vertically aligned.
    '''
    return [(x-1, y), (x-1, y-1), (x, y-1), (x+1, y-1), (x+1, y), (x+1, y+1), (x, y+1), (x-1, y+1)]