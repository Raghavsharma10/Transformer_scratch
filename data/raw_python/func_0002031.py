def set_database_path(dbfolder):
    """Use to write the database path into the config.

    Parameters
    ----------
    dbfolder : str or pathlib.Path
        Path to where pyciss will store the ISS images it downloads and receives.
    """
    configpath = get_configpath()
    try:
        d = get_config()
    except IOError:
        d = configparser.ConfigParser()
        d['pyciss_db'] = {}
    d['pyciss_db']['path'] = dbfolder
    with configpath.open('w') as f:
        d.write(f)
    print("Saved database path into {}.".format(configpath))