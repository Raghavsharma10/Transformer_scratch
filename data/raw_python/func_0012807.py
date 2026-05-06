def openidf(fname, idd=None, epw=None):
    """automatically set idd and open idf file. Uses version from idf to set correct idd
    It will work under the following circumstances:
    
    - the IDF file should have the VERSION object.
    - Needs  the version of EnergyPlus installed that matches the IDF version.
    - Energyplus should be installed in the default location.

    Parameters
    ----------
    fname : str, StringIO or IOBase
        Filepath IDF file,
        File handle of IDF file open to read
        StringIO with IDF contents within
    idd : str, StringIO or IOBase
        This is an optional argument. easyopen will find the IDD without this arg
        Filepath IDD file,
        File handle of IDD file open to read
        StringIO with IDD contents within
    epw : str
        path name to the weather file. This arg is needed to run EneryPlus from eppy.
    """
    import eppy.easyopen as easyopen
    return easyopen.easyopen(fname, idd=idd, epw=epw)