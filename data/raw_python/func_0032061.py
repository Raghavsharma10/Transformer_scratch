def locate_run(output, target, no_newline):
    """
    Print location of RASH related file.
    """
    from .config import ConfigStore
    cfstore = ConfigStore()
    path = getattr(cfstore, "{0}_path".format(target))
    output.write(path)
    if not no_newline:
        output.write("\n")