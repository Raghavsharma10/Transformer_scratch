def write_script(script, tempdir):
    """Write script to a temporary directory

    Arguments:
        script (list): Commands which to put into a file

    Returns:
        Absolute path to script

    """

    name = "script" + self.suffix
    path = os.path.join(tempdir, name)

    with open(path, "w") as f:
        f.write("\n".join(script))

    return path