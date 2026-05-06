def partition_scripts(scripts, start_type1, start_type2):
    """Return two lists of scripts out of the original `scripts` list.

    Scripts that begin with a `start_type1` or `start_type2` blocks 
    are returned first. All other scripts are returned second.

    """
    match, other = [], []
    for script in scripts:
        if (HairballPlugin.script_start_type(script) == start_type1 or 
            HairballPlugin.script_start_type(script) == start_type2):
            match.append(script)
        else:
            other.append(script)
    return match, other