def install_theme(path_to_theme):
    """
    Pass a path to a theme file which will be extracted to the themes directory.
    """
    pref_init()
    # cp the file
    filename = basename(path_to_theme)
    dest = join(THEMES_DIR, filename)
    copy(path_to_theme, dest)
    # unzip
    zf = zipfile.ZipFile(dest)
    # should make sure zipfile contains only themename folder which doesn't conflict
    # with existing themename. Or some kind of sanity check
    zf.extractall(THEMES_DIR)  # plus this is a potential security flaw pre 2.7.4
    # remove the copied zipfile
    unlink(dest)