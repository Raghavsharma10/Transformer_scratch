def value(board, who='x'):
    """Returns the value of a board
    >>> b = Board(); b._rows = [['x', 'x', 'x'], ['x', 'x', 'x'], ['x', 'x', 'x']]
    >>> value(b)
    1
    >>> b = Board(); b._rows = [['o', 'o', 'o'], ['o', 'o', 'o'], ['o', 'o', 'o']]
    >>> value(b)
    -1
    >>> b = Board(); b._rows = [['x', 'o', ' '], ['x', 'o', ' '], [' ', ' ', ' ']]
    >>> value(b)
    1
    >>> b._rows[0][2] = 'x'
    >>> value(b)
    -1
    """
    w = board.winner()
    if w == who:
        return 1
    if w == opp(who):
        return -1
    if board.turn == 9:
        return 0

    if who == board.whose_turn:
        return max([value(b, who) for b in board.possible()])
    else:
        return min([value(b, who) for b in board.possible()])