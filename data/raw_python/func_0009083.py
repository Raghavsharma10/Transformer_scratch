def get_level(level,version=None,include_files=None,skip_files=None):
    '''get_level returns a single level, with option to customize files
    added and skipped.
    '''

    levels = get_levels(version=version)
    level_names = list(levels.keys())

    if level.upper() in level_names:
        level = levels[level]
    else:
        bot.warning("%s is not a valid level. Options are %s"  %(level.upper(),
                                                                "\n".join(levels)))
        return None

    # Add additional files to skip or remove, if defined
    if skip_files is not None:
        level = modify_level(level,'skip_files',skip_files)
    if include_files is not None:
        level = modify_level(level,'include_files',include_files)

    level = make_level_set(level)
    return level