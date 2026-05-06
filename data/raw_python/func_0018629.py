def convert_magicc7_to_openscm_variables(variables, inverse=False):
    """
    Convert MAGICC7 variables to OpenSCM variables

    Parameters
    ----------
    variables : list_like, str
        Variables to convert

    inverse : bool
        If True, convert the other way i.e. convert OpenSCM variables to MAGICC7
        variables

    Returns
    -------
    ``type(variables)``
        Set of converted variables
    """
    if isinstance(variables, (list, pd.Index)):
        return [
            _apply_convert_magicc7_to_openscm_variables(v, inverse) for v in variables
        ]
    else:
        return _apply_convert_magicc7_to_openscm_variables(variables, inverse)