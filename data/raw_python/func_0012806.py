def newidf(version=None):
    """open a new idf file
    
    easy way to open a new idf file for particular version. Works only id Energyplus of that version is installed. 
    
    Parameters
    ----------
    version: string
        version of the new file you want to create. Will work only if this version of Energyplus has been installed. 

    Returns
    -------
    idf
       file of type eppy.modelmake.IDF
    """  # noqa: E501
    if not version:
        version = "8.9"
    import eppy.easyopen as easyopen
    idfstring = "  Version,{};".format(str(version))
    fhandle = StringIO(idfstring)
    return easyopen.easyopen(fhandle)