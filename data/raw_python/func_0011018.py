def is_won(grid):
    "Did the latest move win the game?"
    p, q = grid
    return any(way == (way & q) for way in ways_to_win)