def calc_piece_size(size, min_piece_size=20, max_piece_size=29, max_piece_count=1000):
    """
    Calculates a good piece size for a size
    """
    logger.debug('Calculating piece size for %i' % size)

    for i in range(min_piece_size, max_piece_size): # 20 = 1MB
        if size / (2**i) < max_piece_count:
            break
    return 2**i