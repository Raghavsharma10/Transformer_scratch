def which(fname):
    """Find location of executable."""
    if "PATH" not in os.environ or not os.environ["PATH"]:
        path = os.defpath
    else:
        path = os.environ["PATH"]

    for p in [fname] + [os.path.join(x, fname) for x in path.split(os.pathsep)]:
        p = os.path.abspath(p)
        if os.access(p, os.X_OK) and not os.path.isdir(p):
            return p

    p = sp.Popen("locate %s" % fname, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    (stdout, stderr) = p.communicate()
    if not stderr:
        for p in stdout.decode().split("\n"):
            if (os.path.basename(p) == fname) and (
                os.access(p, os.X_OK)) and (
                not os.path.isdir(p)):
                return p