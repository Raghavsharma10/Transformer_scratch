def write_aliases(aliases, tempdir):
    """Write aliases to temporary directory

    Arguments:
        aliases (dict): {name: value} dict of aliases
        tempdir (str): Absolute path to where aliases will be stored

    """

    platform = lib.platform()
    if platform == "unix":
        home_alias = "cd $BE_DEVELOPMENTDIR"
    else:
        home_alias = "cd %BE_DEVELOPMENTDIR%"

    aliases["home"] = home_alias

    tempdir = os.path.join(tempdir, "aliases")
    os.makedirs(tempdir)

    for alias, cmd in aliases.iteritems():
        path = os.path.join(tempdir, alias)

        if platform == "windows":
            path += ".bat"

        with open(path, "w") as f:
            f.write(cmd)

        if platform == "unix":
            # Make executable
            st = os.stat(path)
            os.chmod(path, st.st_mode | stat.S_IXUSR
                     | stat.S_IXGRP | stat.S_IXOTH)

    return tempdir