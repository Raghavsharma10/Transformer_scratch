def install_paths(version=None, iddname=None):
    """Get the install paths for EnergyPlus executable and weather files.

    We prefer to get the install path from the IDD name but fall back to
    getting it from the version number for backwards compatibility and to
    simplify tests.

    Parameters
    ----------
    version : str, optional
        EnergyPlus version in the format "X-X-X", e.g. "8-7-0".
    iddname : str, optional
        File path to the IDD.

    Returns
    -------
    eplus_exe : str
        Full path to the EnergyPlus executable.
    eplus_weather : str
        Full path to the EnergyPlus weather directory.

    """
    try:
        eplus_exe, eplus_home = paths_from_iddname(iddname)
    except (AttributeError, TypeError, ValueError):
        eplus_exe, eplus_home = paths_from_version(version)
    eplus_weather = os.path.join(eplus_home, 'WeatherData')

    return eplus_exe, eplus_weather