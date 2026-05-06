def is_checkmate(position, input_color):
    """
    Finds if particular King is checkmated.

    :type: position: Board
    :type: input_color: Color
    :rtype: bool
    """
    return position.no_moves(input_color) and \
        position.get_king(input_color).in_check(position)