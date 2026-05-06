def build_blast_db_from_fasta_file(fasta_file, is_protein=False,
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
    output_dir = output_dir or '.'
    _, fasta_path = mkstemp(dir=output_dir, prefix="BLAST_temp_db_",
                            suffix=".fasta")

    fasta_f = open(fasta_path, 'w')
    for line in fasta_file:
        fasta_f.write('%s\n' % line.strip())
    fasta_f.close()

    blast_db, db_filepaths = build_blast_db_from_fasta_path(fasta_path,
                                                            is_protein=is_protein,
                                                            output_dir=None,
                                                            HALT_EXEC=HALT_EXEC
                                                            )

    db_filepaths.append(fasta_path)

    return blast_db, db_filepaths