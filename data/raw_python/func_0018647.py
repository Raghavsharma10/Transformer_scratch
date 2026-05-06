def _get_openscm_var_from_filepath(filepath):
    """
    Determine the OpenSCM variable from a filepath.

    Uses MAGICC's internal, implicit, filenaming conventions.

    Parameters
    ----------
    filepath : str
        Filepath from which to determine the OpenSCM variable.

    Returns
    -------
    str
        The OpenSCM variable implied by the filepath.
    """
    reader = determine_tool(filepath, "reader")(filepath)
    openscm_var = convert_magicc7_to_openscm_variables(
        convert_magicc6_to_magicc7_variables(reader._get_variable_from_filepath())
    )

    return openscm_var