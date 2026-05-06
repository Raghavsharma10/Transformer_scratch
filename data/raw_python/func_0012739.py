def paths_from_version(version):
    """Get the EnergyPlus install directory and executable path.

    Parameters
    ----------
    version : str, optional
        EnergyPlus version in the format "X-X-X", e.g. "8-7-0".

    Returns
    -------
    eplus_exe : str
        Full path to the EnergyPlus executable.
    eplus_home : str
        Full path to the EnergyPlus install directory.

    """
    if platform.system() == 'Windows':
        eplus_home = "C:/EnergyPlusV{version}".format(version=version)
        eplus_exe = os.path.join(eplus_home, 'energyplus.exe')
    elif platform.system() == "Linux":
        eplus_home = "/usr/local/EnergyPlus-{version}".format(version=version)
        eplus_exe = os.path.join(eplus_home, 'energyplus')
    else:
        eplus_home = "/Applications/EnergyPlus-{version}".format(version=version)
        eplus_exe = os.path.join(eplus_home, 'energyplus')
    return eplus_exe, eplus_home