def get_local_filepath(filename):
    """
    Helper for finding our raw SQL files locally.

    Expects files to be in:
        $SOCORRO_PATH/socorrolib/external/postgresql/raw_sql/procs/
    """
    procs_dir = os.path.normpath(os.path.join(
        __file__,
        '../../',
        'external/postgresql/raw_sql/procs'
    ))
    return os.path.join(procs_dir, filename)