def pick_theme(manual):
    """
    Return theme name based on manual input, prefs file, or default to "plain".
    """
    if manual:
        return manual
    pref_init()
    parser = cp.ConfigParser()
    parser.read(PREFS_FILE)
    try:
        theme = parser.get("theme", "default")
    except (cp.NoSectionError, cp.NoOptionError):
        theme = "plain"
    return theme