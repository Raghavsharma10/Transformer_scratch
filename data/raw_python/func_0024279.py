def no_moves(position):
    """
    Finds if the game is over.

    :type: position: Board
    :rtype: bool
    """
    return position.no_moves(color.white) \
        or position.no_moves(color.black)