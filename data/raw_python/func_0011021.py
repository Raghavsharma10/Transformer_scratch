def view(grid):
    "Show a grid human-readably."
    p_mark, q_mark = player_marks(grid)
    return grid_format % tuple(p_mark if by_p else q_mark if by_q else '.'
                               for by_p, by_q in zip(*map(player_bits, grid)))