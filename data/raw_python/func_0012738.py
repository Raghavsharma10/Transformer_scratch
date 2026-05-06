def paths_from_iddname(iddname):
    """Get the EnergyPlus install directory and executable path.

    Parameters
    ----------
    iddname : str, optional
        File path to the IDD.

    Returns
    -------
    eplus_exe : str
        Full path to the EnergyPlus executable.
    eplus_home : str
        Full path to the EnergyPlus install directory.

    Raises
    ------
    AttributeError (TypeError on Windows)
        If iddname does not have a directory component (e.g. if None).
    ValueError
        If eplus_exe is not a file.

    """
    eplus_home = os.path.abspath(os.path.dirname(iddname))
    if platform.system() == 'Windows':
        eplus_exe = os.path.join(eplus_home, 'energyplus.exe')
    elif platform.system() == "Linux":
        eplus_exe = os.path.join(eplus_home, 'energyplus')
    else:
        eplus_exe = os.path.join(eplus_home, 'energyplus')
    if not os.path.isfile(eplus_exe):
        raise ValueError
    return eplus_exe, eplus_home