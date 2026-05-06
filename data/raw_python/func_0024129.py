def incomplete_alg(alg_str, input_color, position):
    """
    Converts a string written in short algebraic form into an incomplete move.
    These incomplete moves do not have the initial location specified and
    therefore cannot be used to update the board. IN order to fully utilize
    incomplete move, it must be run through ``make_legal()`` with
    the corresponding position. It is recommended to use
    ``short_alg()`` instead of this method because it returns a complete
    move.

    Examples: e4, Nf3, exd5, Qxf3, 00, 000, e8=Q

    :type: alg_str: str
    :type: input_color: Color
    """
    edge_rank = 0 \
        if input_color == color.white \
        else 7

    if alg_str is None or len(alg_str) <= 1:
        raise ValueError("algebraic string {} is invalid".format(alg_str))

    # King-side castle
    if alg_str in ["00", "oo", "OO", "0-0", "o-o", "O-O"]:
        return Move(end_loc=Location(edge_rank, 6),
                    piece=King(input_color, Location(edge_rank, 4)),
                    status=notation_const.KING_SIDE_CASTLE,
                    start_loc=Location(edge_rank, 4))

    # Queen-side castle
    if alg_str in ["000", "ooo", "OOO", "0-0-0", "o-o-o", "O-O-O"]:
        return Move(end_loc=Location(edge_rank, 2),
                    piece=King(input_color, Location(edge_rank, 4)),
                    status=notation_const.QUEEN_SIDE_CASTLE,
                    start_loc=Location(edge_rank, 4))
    try:
        end_location = Location.from_string(alg_str[-2:])
    except ValueError:
        end_location = Location.from_string(alg_str[-4:-2])

    # Pawn movement
    if len(alg_str) == 2:
        possible_pawn = position.piece_at_square(end_location.shift_back(input_color))
        if type(possible_pawn) is Pawn and \
                possible_pawn.color == input_color:
            start_location = end_location.shift_back(input_color)
        else:
            start_location = end_location.shift_back(input_color, times=2)
        return Move(end_loc=end_location,
                    piece=position.piece_at_square(start_location),
                    status=notation_const.MOVEMENT,
                    start_loc=start_location)

    # Non-pawn Piece movement
    if len(alg_str) == 3:
        possible_piece, start_location = _get_piece_start_location(end_location,
                                                                   input_color,
                                                                   _get_piece(alg_str, 0),
                                                                   position)
        return Move(end_loc=end_location,
                    piece=possible_piece,
                    status=notation_const.MOVEMENT,
                    start_loc=start_location)

    # Multiple options (Capture or Piece movement with file specified)
    if len(alg_str) == 4:

        # Capture
        if alg_str[1].upper() == "X":

            # Pawn capture
            if not alg_str[0].isupper():
                pawn_location = Location(end_location.rank, ord(alg_str[0]) - 97).shift_back(input_color)
                possible_pawn = position.piece_at_square(pawn_location)
                if type(possible_pawn) is Pawn and \
                        possible_pawn.color == input_color:
                    en_passant_pawn = position.piece_at_square(end_location.shift_back(input_color))
                    if type(en_passant_pawn) is Pawn and \
                            en_passant_pawn.color != input_color and \
                            position.is_square_empty(end_location):
                        return Move(end_loc=end_location,
                                    piece=position.piece_at_square(pawn_location),
                                    status=notation_const.EN_PASSANT,
                                    start_loc=pawn_location)
                    else:
                        return Move(end_loc=end_location,
                                    piece=position.piece_at_square(pawn_location),
                                    status=notation_const.CAPTURE,
                                    start_loc=pawn_location)

            # Piece capture
            elif alg_str[0].isupper():
                possible_piece, start_location = _get_piece_start_location(end_location,
                                                                           input_color,
                                                                           _get_piece(alg_str, 0),
                                                                           position)
                return Move(end_loc=end_location,
                            piece=possible_piece,
                            status=notation_const.CAPTURE,
                            start_loc=start_location)

        # Pawn Promotion
        elif alg_str[2] == "=":
            promote_end_loc = Location.from_string(alg_str[:2])
            if promote_end_loc.rank != 0 and promote_end_loc.rank != 7:
                raise ValueError("Promotion {} must be on the last rank".format(alg_str))
            return Move(end_loc=promote_end_loc,
                        piece=Pawn(input_color, promote_end_loc),
                        status=notation_const.PROMOTE,
                        promoted_to_piece=_get_piece(alg_str, 3),
                        start_loc=promote_end_loc.shift_back(input_color))

        # Non-pawn Piece movement with file specified (aRb7)
        elif alg_str[1].isupper() and not alg_str[0].isdigit():
            possible_piece, start_location = _get_piece_start_location(end_location,
                                                                       input_color,
                                                                       _get_piece(alg_str, 1),
                                                                       position,
                                                                       start_file=alg_str[0])
            return Move(end_loc=end_location,
                        piece=possible_piece,
                        status=notation_const.MOVEMENT,
                        start_loc=start_location)

        # (alt) Non-pawn Piece movement with file specified (Rab7)
        elif alg_str[0].isupper() and not alg_str[1].isdigit():
            possible_piece, start_location = _get_piece_start_location(end_location,
                                                                       input_color,
                                                                       _get_piece(alg_str, 0),
                                                                       position,
                                                                       start_file=alg_str[1])
            return Move(end_loc=end_location,
                        piece=possible_piece,
                        status=notation_const.MOVEMENT,
                        start_loc=start_location)

        # Non-pawn Piece movement with rank specified (R1b7)
        elif alg_str[0].isupper() and alg_str[1].isdigit():
            possible_piece, start_location = _get_piece_start_location(end_location,
                                                                       input_color,
                                                                       _get_piece(alg_str, 0),
                                                                       position,
                                                                       start_rank=alg_str[1])
            return Move(end_loc=end_location,
                        piece=possible_piece,
                        status=notation_const.MOVEMENT,
                        start_loc=start_location)

    # Multiple options
    if len(alg_str) == 5:

        # Non-pawn Piece movement with rank and file specified (a2Ra1
        if not alg_str[0].isdigit() and \
                alg_str[1].isdigit() and \
                alg_str[2].isupper() and \
                not alg_str[3].isdigit() and \
                alg_str[4].isdigit:
            start_loc = Location.from_string(alg_str[:2])
            return Move(end_loc=end_location,
                        piece=_get_piece(alg_str, 2)(input_color, end_location),
                        status=notation_const.MOVEMENT,
                        start_loc=start_loc)

        # Multiple Piece capture options
        if alg_str[2].upper() == "X":

            # Piece capture with rank specified (R1xa1)
            if alg_str[1].isdigit():
                possible_piece, start_location = _get_piece_start_location(end_location,
                                                                           input_color,
                                                                           _get_piece(alg_str, 0),
                                                                           position,
                                                                           start_rank=alg_str[1])
                return Move(end_loc=end_location,
                            piece=possible_piece,
                            status=notation_const.CAPTURE,
                            start_loc=start_location)

            # Piece capture with file specified (Rdxd7)
            else:
                possible_piece, start_location = _get_piece_start_location(end_location,
                                                                           input_color,
                                                                           _get_piece(alg_str, 0),
                                                                           position,
                                                                           start_file=alg_str[1])
                return Move(end_loc=end_location,
                            piece=possible_piece,
                            status=notation_const.CAPTURE,
                            start_loc=start_location)

    # Pawn promotion with capture
    if len(alg_str) == 6 and alg_str[4] == "=":
        start_file = ord(alg_str[0]) - 97
        promote_capture_end_loc = Location.from_string(alg_str[2:4])
        return Move(end_loc=promote_capture_end_loc,
                    piece=Pawn(input_color, promote_capture_end_loc),
                    status=notation_const.CAPTURE_AND_PROMOTE,
                    promoted_to_piece=_get_piece(alg_str, 5),
                    start_loc=Location(end_location.shift_back(input_color).rank, start_file))

    raise ValueError("algebraic string {} is invalid in \n{}".format(alg_str, position))