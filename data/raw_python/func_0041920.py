def paste(cmd=paste_cmd, stdout=PIPE):
    """Returns system clipboard contents.
    """
    return Popen(cmd, stdout=stdout).communicate()[0].decode('utf-8')