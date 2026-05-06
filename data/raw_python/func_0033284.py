def build_blast_db_from_seqs(seqs, is_protein=False, output_dir='./',
                             HALT_EXEC=False):
    """Build blast db from seqs; return db name and list of files created

        **If using to create temporary blast databases, you can call
        cogent.util.misc.remove_files(db_filepaths) to clean up all the
        files created by formatdb when you're done with the database.

        seqs: sequence collection or alignment object
        is_protein: True if working on protein seqs (default: False)
        output_dir: directory where output should be written
         (default: current directory)
        HALT_EXEC: halt just before running the formatdb command and
         print the command -- useful for debugging
    """

    # Build a temp filepath
    _, tmp_fasta_filepath = mkstemp(prefix='Blast_tmp_db', suffix='.fasta')
    # open the temp file
    tmp_fasta_file = open(tmp_fasta_filepath, 'w')
    # write the sequence collection to file
    tmp_fasta_file.write(seqs.toFasta())
    tmp_fasta_file.close()

    # build the bast database
    db_name, db_filepaths = build_blast_db_from_fasta_path(tmp_fasta_filepath,
                                                           is_protein=is_protein,
                                                           output_dir=output_dir,
                                                           HALT_EXEC=HALT_EXEC)

    # clean-up the temporary file
    remove(tmp_fasta_filepath)

    # return the results
    return db_name, db_filepaths