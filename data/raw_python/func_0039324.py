def cowsay(text='', align='centre'):
    """
    Simulate an ASCII cow saying text.

    :type text: string
    :param text: The text to print out.

    :type align: string
    :param algin: Where to align the cow. Can be 'left', 'centre' or 'right'
    """

    # Make align lowercase
    align = align.lower()

    # Set the cowtext
    cowtext = str(text)

    # Set top part of speech bubble to the length of the text plus 2
    topbar = ' ' * (len(text) + 2)

    # Set bottom part of speech bubble to the length of the text plus 2
    bottombar = ' ' * (len(text) + 2)

    # If align is centre
    if align in ["center", "centre"]:
        # Set the spacing before the cow to the length of half of the length of topbar plus 1
        spacing = " " * (int(len(topbar) / 2) + 1)

    # If align is left
    elif align == 'left':
        # Set spacing to a single space
        spacing = ' '

    # If align is right
    elif align == 'right':
        # Set the spacing to the length of the text plus 2
        spacing = " " * (len(text) + 2)

    else:
        # Raise a runtime warning
        raise ValueError("Invalid alignment provided.")

    # Print the top bar
    print(topbar)

    # Print the text
    print('( ' + repr(str(cowtext)) + ' )')

    # Print the bottom bar
    print(bottombar)

    # Print the cow with the spacing
    print(spacing + r'o   ^__^ ')
    print(spacing + r' o  (oO)\_______')
    print(spacing + r'    (__)\       )\/\ ')
    print(spacing + r'     U  ||----w | ')
    print(spacing + r'        ||     || ')