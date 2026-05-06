def get_levels(version=None):
    '''get_levels returns a dictionary of levels (key) and values (dictionaries with
    descriptions and regular expressions for files) for the user. 
    :param version: the version of singularity to use (default is 2.2)
    :param include_files: files to add to the level, only relvant if
    '''
    valid_versions = ['2.3','2.2']

    if version is None:
        version = "2.3"  
    version = str(version)

    if version not in valid_versions:
        bot.error("Unsupported version %s, valid versions are %s" %(version,
                                                                    ",".join(valid_versions)))

    levels_file = os.path.abspath(os.path.join(get_installdir(),
                                                           'analysis',
                                                           'reproduce',
                                                           'data',
                                                           'reproduce_levels.json'))
    levels = read_json(levels_file)
    if version == "2.2":
        # Labels not added until 2.3
        del levels['LABELS']

    levels = make_levels_set(levels)

    return levels