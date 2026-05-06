def get_terminal_size():
    """
    get size of console: rows x columns

    :return: tuple, (int, int)
    """
    try:
        rows, columns = subprocess.check_output(['stty', 'size']).split()
    except subprocess.CalledProcessError:
        # not attached to terminal
        logger.info("not attached to terminal")
        return 0, 0
    logger.debug("console size is %s %s", rows, columns)
    return int(rows), int(columns)