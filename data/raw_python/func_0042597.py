def cell_strings(term):
    """Return the strings that represent each possible living cell state.

    Return the most colorful ones the terminal supports.

    """
    num_colors = term.number_of_colors
    if num_colors >= 16:
        funcs = term.on_bright_red, term.on_bright_green, term.on_bright_cyan
    elif num_colors >= 8:
        funcs = term.on_red, term.on_green, term.on_blue
    else:
        # For black and white, use the checkerboard cursor from the vt100
        # alternate charset:
        return (term.reverse(' '),
                term.smacs + term.reverse('a') + term.rmacs,
                term.smacs + 'a' + term.rmacs)
    # Wrap spaces in whatever pretty colors we chose:
    return [f(' ') for f in funcs]