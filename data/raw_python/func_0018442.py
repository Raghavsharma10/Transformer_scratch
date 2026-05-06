def name2rgb(name):
    """Convert the name of a color into its RGB value"""
    try:
        import colour
    except ImportError:
        raise ImportError('You need colour to be installed: pip install colour')

    c = colour.Color(name)
    color = int(c.red * 255), int(c.green * 255), int(c.blue * 255)
    return color