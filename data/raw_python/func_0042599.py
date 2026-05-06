def next_board(board, wrap):
    """Given a board, return the board one interation later.

    Adapted from Jack Diedrich's implementation from his 2012 PyCon talk "Stop
    Writing Classes"

    :arg wrap: A callable which takes a point and transforms it, for example
        to wrap to the other edge of the screen. Return None to remove a point.

    """
    new_board = {}

    # We need consider only the points that are alive and their neighbors:
    points_to_recalc = set(board.iterkeys()) | set(chain(*map(neighbors, board)))

    for point in points_to_recalc:
        count = sum((neigh in board) for neigh in
                    (wrap(n) for n in neighbors(point) if n))
        if count == 3:
            state = 0 if point in board else 1
        elif count == 2 and point in board:
            state = 2
        else:
            state = None

        if state is not None:
            wrapped = wrap(point)
            if wrapped:
                new_board[wrapped] = state

    return new_board