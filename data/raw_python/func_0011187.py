def _bashcomplete(cmd, prog_name, complete_var=None):
    """Internal handler for the bash completion support."""
    if complete_var is None:
        complete_var = '_%s_COMPLETE' % (prog_name.replace('-', '_')).upper()
    complete_instr = os.environ.get(complete_var)
    if not complete_instr:
        return

    from ._bashcomplete import bashcomplete
    if bashcomplete(cmd, prog_name, complete_var, complete_instr):
        sys.exit(1)