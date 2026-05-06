def print_page(text):
    """Format the text and prints it on stdout.

    Text is formatted by adding a ASCII frame around it and coloring the text.
    Colors can be added to text using color tags, for example:

        My [FG_BLUE]blue[NORMAL] text.
        My [BG_BLUE]blue background[NORMAL] text.
    """
    color_re = re.compile(r"\[(?P<color>[FB]G_[A-Z_]+|NORMAL)\]")
    width = max([len(strip_colors(x)) for x in text.splitlines()])
    print("\n" + hbar(width))
    for line in text.splitlines():
        if line == "[HBAR]":
            print(hbar(width))
            continue
        tail = width - len(strip_colors(line))
        sys.stdout.write("| ")
        previous = 0
        end = len(line)
        for match in color_re.finditer(line):
            sys.stdout.write(line[previous : match.start()])
            set_color(match.groupdict()["color"])
            previous = match.end()
        sys.stdout.write(line[previous:end])
        sys.stdout.write(" " * tail + " |\n")
    print(hbar(width))