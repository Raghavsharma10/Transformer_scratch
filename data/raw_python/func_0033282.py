def build_blast_db_from_fasta_path(fasta_path, is_protein=False,
                                   output_dir=None, HALT_EXEC=False):
    """Build blast db from fasta_path; return db name and list of files created

        **If using to create temporary blast databases, you can call
        cogent.util.misc.remove_files(db_filepaths) to clean up all the
        files created by formatdb when you're done with the database.

        fasta_path: path to fasta file of sequences to build database from
        is_protein: True if working on protein seqs (default: False)
        output_dir: directory where output should be written
         (default: directory containing fasta_path)
        HALT_EXEC: halt just before running the formatdb command and
         print the command -- useful for debugging
    """
    fasta_dir, fasta_filename = split(fasta_path)
    if not output_dir:
        output_dir = fasta_dir or '.'
        # Will cd to this directory, so just pass the filename
        # so the app is not confused by relative paths
        fasta_path = fasta_filename

    if not output_dir.endswith('/'):
        db_name = output_dir + '/' + fasta_filename
    else:
        db_name = output_dir + fasta_filename

    # instantiate the object
    fdb = FormatDb(WorkingDir=output_dir, HALT_EXEC=HALT_EXEC)
    if is_protein:
        fdb.Parameters['-p'].on('T')
    else:
        fdb.Parameters['-p'].on('F')
    app_result = fdb(fasta_path)
    db_filepaths = []
    for v in app_result.values():
        try:
            db_filepaths.append(v.name)
        except AttributeError:
            # not a file object, so no path to return
            pass
    return db_name, db_filepaths