def get_slanted_adjacent_coords(x, y):
    '''
    returns the six adjacent coordinates on a standard keyboard, where each row is slanted to the right from the last.
    adjacencies are clockwise, starting with key to the left, then two keys above, then right key, then two keys below.
    (that is, only near-diagonal keys are adjacent, so g's coordinate is adjacent to those of t,y,b,v, but not those of r,u,n,c.)
    '''
    return [(x-1, y), (x, y-1), (x+1, y-1), (x+1, y), (x, y+1), (x-1, y+1)]