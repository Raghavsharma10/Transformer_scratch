def cprint(
    text,
    fg=Color.normal,
    bg=Color.normal,
    fg_dark=False,
    bg_dark=False,
    underlined=False,
    parse=False,
):
    """Print string in to stdout using colored font.

    See L{set_color} for more details about colors.

    Args:
        text (str): Text that needs to be printed.
    """
    if parse:
        color_re = Color.color_re()
        lines = text.splitlines()
        count = len(lines)
        for i, line in enumerate(lines):
            previous = 0
            end = len(line)
            for match in color_re.finditer(line):
                sys.stdout.write(line[previous : match.start()])
                d = match.groupdict()
                set_color(
                    d["color"], fg_dark=False if d["dark"] is None else True
                )
                previous = match.end()
            sys.stdout.write(
                line[previous:end]
                + ("\n" if (i < (count - 1) or text[-1] == "\n") else "")
            )
    else:
        set_color(fg, bg, fg_dark, bg_dark, underlined)
        sys.stdout.write(text)
        set_color()