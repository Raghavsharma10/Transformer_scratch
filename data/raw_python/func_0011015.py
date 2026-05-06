def max_play(w, i, grid):
    "Play like Spock, except breaking ties by drunk_value."
    return min(successors(grid),
               key=lambda succ: (evaluate(succ), drunk_value(succ)))