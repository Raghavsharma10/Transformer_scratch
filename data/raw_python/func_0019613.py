def locate_tool(name, verbose=True): 
    """
    Returns the binary of a tool.

    Parameters
    ----------
    name : str
        Name of the tool (case-insensitive).

    Returns
    -------
    tool_bin : str
        Binary of tool.
    """
    m = get_tool(name) 
    tool_bin = which(m.cmd) 
    if tool_bin:
        if verbose:
            print("Found {} in {}".format(m.name, tool_bin)) 
        return tool_bin 
    else: 
        print("Couldn't find {}".format(m.name))