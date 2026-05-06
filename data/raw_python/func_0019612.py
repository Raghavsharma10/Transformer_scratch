def get_tool(name): 
    """
    Returns an instance of a specific tool.

    Parameters
    ----------
    name : str
        Name of the tool (case-insensitive).

    Returns
    -------
    tool : MotifProgram instance
    """
    tool = name.lower()
    if tool not in __tools__:
        raise ValueError("Tool {0} not found!\n".format(name))

    t = __tools__[tool]()

    if not t.is_installed():
        sys.stderr.write("Tool {0} not installed!\n".format(tool))

    if not t.is_configured():
        sys.stderr.write("Tool {0} not configured!\n".format(tool))

    return t