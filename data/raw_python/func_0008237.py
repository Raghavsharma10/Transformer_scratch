def addStyle(w):
    """
    Styles the GUI: global fonts and colours.

    Parameters
    ----------
    w : tkinter.tk
        widget element to style
    """
    # access global container in root widget
    root = get_root(w)
    g = root.globals
    fsize = g.cpars['font_size']
    family = g.cpars['font_family']

    # Default font
    g.DEFAULT_FONT = font.nametofont("TkDefaultFont")
    g.DEFAULT_FONT.configure(size=fsize, weight='bold', family=family)
    w.option_add('*Font', g.DEFAULT_FONT)

    # Menu font
    g.MENU_FONT = font.nametofont("TkMenuFont")
    g.MENU_FONT.configure(family=family)
    w.option_add('*Menu.Font', g.MENU_FONT)

    # Entry font
    g.ENTRY_FONT = font.nametofont("TkTextFont")
    g.ENTRY_FONT.configure(size=fsize, family=family)
    w.option_add('*Entry.Font', g.ENTRY_FONT)

    # position and size
    # root.geometry("320x240+325+200")

    # Default colours. Note there is a difference between
    # specifying 'background' with a capital B or lowercase b
    w.option_add('*background', g.COL['main'])
    w.option_add('*HighlightBackground', g.COL['main'])
    w.config(background=g.COL['main'])