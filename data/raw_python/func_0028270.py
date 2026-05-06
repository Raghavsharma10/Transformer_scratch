def print_title(title, is_end=False):
    """
    Print title like ``----- {title} -----`` or ``===== {title} =====``.

    :param title: Title.

    :param is_end: Whether is end title. End title use ``=`` instead of ``-``.

    :return: None.
    """
    # If is end title
    if is_end:
        # Use `=`
        sep = '====='

    # If is not end title
    else:
        # Use `-`
        sep = '-----'

    # If is not end title
    if not is_end:
        # Print an empty line for visual comfort
        print_text()

    # Print the title, e.g. `----- {title} -----`
    print_text('# {sep} {title} {sep}'.format(title=title, sep=sep))