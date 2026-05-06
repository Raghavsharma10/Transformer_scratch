def player_marks(grid):
    "Return two results: the player's mark and their opponent's."
    p, q = grid
    return 'XO' if sum(player_bits(p)) == sum(player_bits(q)) else 'OX'