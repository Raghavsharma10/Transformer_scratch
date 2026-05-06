def colorize_output(output, colors, indent=0):
    r"""Print output to console using provided color mappings.

    Color mapping is dict with regular expressions as key and tuple of two as
    values. Key is used to match if line should be colorized and tuple contains
    color to be used and boolean value that indicates if dark foreground
    is used.
    For example:

        >>> CLS = {
        >>>     re.compile(r'^(--- .*)$'): (Color.red, False)
        >>> }

    will colorize lines that start with '---' to red.

    If different parts of line needs to be in different color then dict must be
    supplied in colors with keys that are named group from regular expression
    and values that are tuples of color and boolean that indicates if dark
    foreground is used.
    For example:

        >>> CLS = {
        >>>     re.compile(r'^(?P<key>user:\s+)(?P<user>.*)$'): {
        >>>         'key': (Color.yellow, True),
        >>>         'user': (Color.cyan,   False)
        >>>     }
        >>> }

    will colorize line 'user: Some user' so that 'user:' part is yellow with
    dark foreground and 'Some user' part is cyan without dark foreground.
    """
    for line in output.split("\n"):
        cprint(" " * indent)
        if line == "":
            cprint("\n")
            continue
        for regexp, color_def in colors.items():
            if regexp.match(line) is not None:
                _colorize_single_line(line, regexp, color_def)
                break
        else:
            cprint("%s\n" % line)