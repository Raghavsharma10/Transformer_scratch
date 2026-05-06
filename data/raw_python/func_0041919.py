def copy(string, cmd=copy_cmd, stdin=PIPE):
    """Copy given string into system clipboard.
    """
    Popen(cmd, stdin=stdin).communicate(string.encode('utf-8'))