def load_stored_proc(op, filelist):
    """
    Takes the alembic op object as arguments and a list of files as arguments
    Load and run CREATE OR REPLACE function commands from files
    """
    for filename in filelist:
        sqlfile = get_local_filepath(filename)
        # Capturing "file not exists" here rather than allowing
        # an exception to be thrown. Some of the rollback scripts
        # would otherwise throw unhelpful exceptions when a SQL
        # file is removed from the repo.
        if not os.path.isfile(sqlfile):
            warnings.warn(
                "Did not find %r. Continuing migration." % sqlfile,
                UserWarning,
                2
            )
            continue
        with open(sqlfile, 'r') as stored_proc:
            op.execute(stored_proc.read())