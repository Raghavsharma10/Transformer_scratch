def assign_dna_reads_to_protein_database(query_fasta_fp, database_fasta_fp,
                                         output_fp, temp_dir="/tmp", params=None):
    """Assign DNA reads to a database fasta of protein sequences.

    Wraps assign_reads_to_database, setting database and query types. All
    parameters are set to default unless params is passed. A temporary
    file must be written containing the translated sequences from the input
    query fasta file because BLAT cannot do this automatically.

    query_fasta_fp: absolute path to the query fasta file containing DNA
                   sequences.
    database_fasta_fp: absolute path to the database fasta file containing
                      protein sequences.
    output_fp: absolute path where the output file will be generated.
    temp_dir: optional. Change the location where the translated sequences
              will be written before being used as the query. Defaults to
              /tmp.
    params: optional. dict containing parameter settings to be used
                  instead of default values. Cannot change database or query
                  file types from protein and dna, respectively.

    This method returns an open file object. The output format
    defaults to blast9 and should be parsable by the PyCogent BLAST parsers.
    """
    if params is None:
        params = {}

    my_params = {'-t': 'prot', '-q': 'prot'}

    # make sure temp_dir specifies an absolute path
    if not isabs(temp_dir):
        raise ApplicationError("temp_dir must be an absolute path.")

    # if the user specified parameters other than default, then use them.
    # However, if they try to change the database or query types, raise an
    # applciation error.
    if '-t' in params or '-q' in params:
        raise ApplicationError("Cannot change database or query types "
                               "when using assign_dna_reads_to_dna_database. Use "
                               "assign_reads_to_database instead.")

    if 'genetic_code' in params:
        my_genetic_code = GeneticCodes[params['genetic_code']]
        del params['genetic_code']
    else:
        my_genetic_code = GeneticCodes[1]

    my_params.update(params)

    # get six-frame translation of the input DNA sequences and write them to
    # temporary file.
    _, tmp = mkstemp(dir=temp_dir)
    tmp_out = open(tmp, 'w')

    for label, sequence in parse_fasta(open(query_fasta_fp)):
        seq_id = label.split()[0]

        s = DNA.makeSequence(sequence)
        translations = my_genetic_code.sixframes(s)
        frames = [1, 2, 3, -1, -2, -3]
        translations = dict(zip(frames, translations))

        for frame, translation in sorted(translations.iteritems()):
            entry = '>{seq_id}_frame_{frame}\n{trans}\n'
            entry = entry.format(seq_id=seq_id, frame=frame, trans=translation)
            tmp_out.write(entry)

    tmp_out.close()
    result = assign_reads_to_database(tmp, database_fasta_fp, output_fp,
                                      params=my_params)

    remove(tmp)

    return result