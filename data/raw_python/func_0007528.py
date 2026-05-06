def use_theme(theme):
    """Make the given theme current.

    There are two included themes: light_theme, dark_theme.
    """
    global current
    current = theme
    import scene
    if scene.current is not None:
        scene.current.stylize()