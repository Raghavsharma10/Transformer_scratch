def Font(name=None, source="sys", italic=False, bold=False, size=20):
    """Unifies loading of fonts.

    :param name: name of system-font or filepath, if None is passed the default
        system-font is loaded

    :type name: str
    :param source: "sys" for system font, or "file" to load a file
    :type source: str
    """
    assert source in ["sys", "file"]
    if not name:
        return pygame.font.SysFont(pygame.font.get_default_font(), 
        size, bold=bold, italic=italic)
    if source == "sys":
        return pygame.font.SysFont(name, 
        size, bold=bold, italic=italic)
    else:
        f = pygame.font.Font(name, size)
        f.set_italic(italic)
        f.set_bold(bold)
        return f