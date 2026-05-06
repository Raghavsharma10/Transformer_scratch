def convert_magicc6_to_magicc7_variables(variables, inverse=False):
    """
    Convert MAGICC6 variables to MAGICC7 variables

    Parameters
    ----------
    variables : list_like, str
        Variables to convert

    inverse : bool
        If True, convert the other way i.e. convert MAGICC7 variables to MAGICC6
        variables

    Raises
    ------
    ValueError
        If you try to convert HFC245ca, or some variant thereof, you will get a
        ValueError. The reason is that this variable was never meant to be included in
        MAGICC6, it was just an accident. See, for example, the text in the
        description section of ``pymagicc/MAGICC6/run/HISTRCP_HFC245fa_CONC.IN``:
        "...HFC245fa, rather than HFC245ca, is the actually used isomer.".

    Returns
    -------
    ``type(variables)``
        Set of converted variables
    """
    if isinstance(variables, (list, pd.Index)):
        return [
            _apply_convert_magicc6_to_magicc7_variables(v, inverse) for v in variables
        ]
    else:
        return _apply_convert_magicc6_to_magicc7_variables(variables, inverse)