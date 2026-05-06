def set_default_theme(theme):
    """
    Set default theme name based in config file.
    """
    pref_init()  # make sure config files exist
    parser = cp.ConfigParser()
    parser.read(PREFS_FILE)
    # Do we need to create a section?
    if not parser.has_section("theme"):
        parser.add_section("theme")
    parser.set("theme", "default", theme)
    # best way to make sure no file truncation?
    with open("%s.2" % PREFS_FILE, "w") as fp:
        parser.write(fp)
    copy("%s.2" % PREFS_FILE, PREFS_FILE)
    unlink("%s.2" % PREFS_FILE,)