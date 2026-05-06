def _get_piece(string, index):
    """
    Returns Piece subclass given index of piece.

    :type: index: int
    :type: loc Location

    :raise: KeyError
    """
    piece = string[index].strip()
    piece = piece.upper()
    piece_dict = {'R': Rook,
                  'P': Pawn,
                  'B': Bishop,
                  'N': Knight,
                  'Q': Queen,
                  'K': King}
    try:
        return piece_dict[piece]
    except KeyError:
        raise ValueError("Piece {} is invalid".format(piece))