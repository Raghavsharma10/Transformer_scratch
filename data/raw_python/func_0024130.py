def make_legal(move, position):
    """
    Converts an incomplete move (initial ``Location`` not specified)
    and the corresponding position into the a complete move
    with the most likely starting point specified. If no moves match, ``None``
    is returned.

    :type: move: Move
    :type: position: Board
    :rtype: Move
    """
    assert isinstance(move, Move)
    for legal_move in position.all_possible_moves(move.color):

        if move.status == notation_const.LONG_ALG:
            if move.end_loc == legal_move.end_loc and \
                    move.start_loc == legal_move.start_loc:
                return legal_move

        elif move == legal_move:
            return legal_move

    raise ValueError("Move {} not legal in \n{}".format(repr(move), position))