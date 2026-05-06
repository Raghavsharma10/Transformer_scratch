def line(separator="-·-", color=None, padding=None, num=1):
    """ Prints a line separator the full width of the terminal.

        @separator: the #str chars to create the line from
        @color: line color from :mod:vital.debug.colors
        @padding: adds extra lines to either the top, bottom or both
            of the line via :func:padd
        @num: #int number of lines to print
        ..
            from vital.debug import line
            line("__")
            ____________________________________________________________________
        ..
    """
    for x in range(num):
        columns = get_terminal_width()
        separator = "".join(
            separator for x in
            range(floor(columns/len(separator))))
        print(padd(colorize(separator.strip(), color), padding))