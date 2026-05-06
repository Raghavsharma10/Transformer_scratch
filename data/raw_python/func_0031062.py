def splitNames(names):
    """
    Split a sequence id string like "Protein name [pathogen name]" into two
    pieces using the final square brackets to delimit the pathogen name.

    @param names: A C{str} "protein name [pathogen name]" string.
    @return: A 2-C{tuple} giving the C{str} protein name and C{str} pathogen
        name. If C{names} cannot be split on square brackets, it is
        returned as the first tuple element, followed by _NO_PATHOGEN_NAME.
    """
    match = _PATHOGEN_RE.match(names)
    if match:
        proteinName = match.group(1).strip()
        pathogenName = match.group(2).strip()
    else:
        proteinName = names
        pathogenName = _NO_PATHOGEN_NAME

    return proteinName, pathogenName