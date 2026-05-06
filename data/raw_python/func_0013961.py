def list_files(tag=None, sat_id=None, data_path=None, format_str=None):
    """Produce a list of ICON EUV files.

    Notes
    -----
    Currently fixed to level-2

    """

    desc = None
    level = tag
    if level == 'level_1':
        code = 'L1'
        desc = None
    elif level == 'level_2':
        code = 'L2'
        desc = None
    else:
        raise ValueError('Unsupported level supplied: ' + level)

    if format_str is None:
        format_str = 'ICON_'+code+'_EUV_Daytime'
        if desc is not None:
            format_str += '_' + desc +'_'
        format_str += '_{year:4d}-{month:02d}-{day:02d}_v{version:02d}r{revision:03d}.NC'

    return pysat.Files.from_os(data_path=data_path,
                                format_str=format_str)