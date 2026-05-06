def build_database_sortmerna(fasta_path,
                             max_pos=None,
                             output_dir=None,
                             temp_dir=tempfile.gettempdir(),
                             HALT_EXEC=False):
    """ Build sortmerna db from fasta_path; return db name
        and list of files created

        Parameters
        ----------
        fasta_path : string
            path to fasta file of sequences to build database.
        max_pos : integer, optional
            maximum positions to store per seed in index
            [default: 10000].
        output_dir : string, optional
            directory where output should be written
            [default: same directory as fasta_path]
        HALT_EXEC : boolean, optional
            halt just before running the indexdb_rna command
            and print the command -- useful for debugging
            [default: False].

        Return
        ------
        db_name : string
            filepath to indexed database.
        db_filepaths : list
            output files by indexdb_rna
    """

    if fasta_path is None:
        raise ValueError("Error: path to fasta reference "
                         "sequences must exist.")

    fasta_dir, fasta_filename = split(fasta_path)
    if not output_dir:
        output_dir = fasta_dir or '.'
        # Will cd to this directory, so just pass the filename
        # so the app is not confused by relative paths
        fasta_path = fasta_filename

    index_basename = splitext(fasta_filename)[0]

    db_name = join(output_dir, index_basename)

    # Instantiate the object
    sdb = IndexDB(WorkingDir=output_dir, HALT_EXEC=HALT_EXEC)

    # The parameter --ref STRING must follow the format where
    # STRING = /path/to/ref.fasta,/path/to/ref.idx
    sdb.Parameters['--ref'].on("%s,%s" % (fasta_path, db_name))

    # Set temporary directory
    sdb.Parameters['--tmpdir'].on(temp_dir)

    # Override --max_pos parameter
    if max_pos is not None:
        sdb.Parameters['--max_pos'].on(max_pos)

    # Run indexdb_rna
    app_result = sdb()

    # Return all output files (by indexdb_rna) as a list,
    # first however remove the StdErr and StdOut filepaths
    # as they files will be destroyed at the exit from
    # this function (IndexDB is a local instance)
    db_filepaths = [v.name for k, v in app_result.items()
                    if k not in {'StdErr', 'StdOut'} and hasattr(v, 'name')]

    return db_name, db_filepaths