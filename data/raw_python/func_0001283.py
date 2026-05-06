def find(name, arg=None):
    """Find process by name or by argument in command line.

    Args:
        name (str): Process name to search for.
        arg (str): Command line argument for a process to search for.

    Returns:
        tea.process.base.IProcess: Process object if found.
    """
    for p in get_processes():
        if p.name.lower().find(name.lower()) != -1:
            if arg is not None:
                for a in p.cmdline or []:
                    if a.lower().find(arg.lower()) != -1:
                        return p
            else:
                return p
    return None