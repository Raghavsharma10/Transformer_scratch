def apply_move(grid, move):
    "Try to move: return a new grid, or None if illegal."
    p, q = grid
    bit = 1 << move
    return (q, p | bit) if 0 == (bit & (p | q)) else None