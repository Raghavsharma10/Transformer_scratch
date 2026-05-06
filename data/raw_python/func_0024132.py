def long_alg(alg_str, position):
    """
    Converts a string written in long algebraic form
    and the corresponding position into a complete move
    (initial location specified). Used primarily for
    UCI, but can be used for other purposes.

    :type: alg_str: str
    :type: position: Board
    :rtype: Move
    """
    if alg_str is None or len(alg_str) < 4 or len(alg_str) > 6:
        raise ValueError("Invalid string input {}".format(alg_str))

    end = Location.from_string(alg_str[2:])
    start = Location.from_string(alg_str[:2])
    piece = position.piece_at_square(start)

    if len(alg_str) == 4:
        return make_legal(Move(end_loc=end,
                               piece=piece,
                               status=notation_const.LONG_ALG,
                               start_loc=start), position)

    promoted_to = _get_piece(alg_str, 4)
    if promoted_to is None or \
            promoted_to is King or \
            promoted_to is Pawn:
        raise Exception("Invalid move input")

    return make_legal(Move(end_loc=end,
                           piece=piece,
                           status=notation_const.LONG_ALG,
                           start_loc=start,
                           promoted_to_piece=promoted_to), position)