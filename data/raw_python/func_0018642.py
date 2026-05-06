def pull_cfg_from_parameters_out(parameters_out, namelist_to_read="nml_allcfgs"):
    """Pull out a single config set from a parameters_out namelist.

    This function returns a single file with the config that needs to be passed to
    MAGICC in order to do the same run as is represented by the values in
    ``parameters_out``.

    Parameters
    ----------
    parameters_out : dict, f90nml.Namelist
        The parameters to dump

    namelist_to_read : str
        The namelist to read from the file.

    Returns
    -------
    :obj:`f90nml.Namelist`
        An f90nml object with the cleaned, read out config.

    Examples
    --------
    >>> cfg = pull_cfg_from_parameters_out(magicc.metadata["parameters"])
    >>> cfg.write("/somewhere/else/ANOTHERNAME.cfg")
    """
    single_cfg = Namelist({namelist_to_read: {}})
    for key, value in parameters_out[namelist_to_read].items():
        if "file_tuning" in key:
            single_cfg[namelist_to_read][key] = ""
        else:
            try:
                if isinstance(value, str):
                    single_cfg[namelist_to_read][key] = value.strip(" \t\n\r").replace(
                        "\x00", ""
                    )
                elif isinstance(value, list):
                    clean_list = [v.strip(" \t\n\r").replace("\x00", "") for v in value]
                    single_cfg[namelist_to_read][key] = [v for v in clean_list if v]
                else:
                    assert isinstance(value, Number)
                    single_cfg[namelist_to_read][key] = value
            except AttributeError:
                if isinstance(value, list):
                    assert all([isinstance(v, Number) for v in value])
                    single_cfg[namelist_to_read][key] = value
                else:
                    raise AssertionError(
                        "Unexpected cause in out parameters conversion"
                    )

    return single_cfg