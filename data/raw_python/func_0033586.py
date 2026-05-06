def assign_reads_to_database(query, database_fasta, out_path, params=None):
    """Assign a set of query sequences to a reference database

    database_fasta_fp: absolute file path to the reference database
    query_fasta_fp: absolute file path to query sequences
    output_fp: absolute file path of the file to be output
    params: dict of BWA specific parameters.
            * Specify which algorithm to use (bwa-short or bwasw) using the
            dict key "algorithm"
            * if algorithm is bwasw, specify params for the bwa bwasw
            subcommand
            * if algorithm is bwa-short, specify params for the bwa samse
            subcommand
            * if algorithm is bwa-short, must also specify params to use with
            bwa aln, which is used to get the sai file necessary to run samse.
            bwa aln params should be passed in using dict key "aln_params" and
            the associated value should be a dict of params for the bwa aln
            subcommand
            * if a temporary directory is not specified in params using dict
            key "temp_dir", it will be assumed to be /tmp

    This method returns an open file object (SAM format).
    """
    if params is None:
        params = {}

    # set the output path
    params['-f'] = out_path

    # if the algorithm is not specified in the params dict, or the algorithm
    # is not recognized, raise an exception
    if 'algorithm' not in params:
        raise InvalidArgumentApplicationError("Must specify which algorithm to"
                                              " use ('bwa-short' or 'bwasw')")
    elif params['algorithm'] not in ('bwa-short', 'bwasw'):
        raise InvalidArgumentApplicationError("Unknown algorithm '%s' Please "
                                              "enter either 'bwa-short' or "
                                              "'bwasw'." % params['algorithm'])

    # if the temp directory is not specified, assume /tmp
    if 'temp_dir' not in params:
        params['temp_dir'] = '/tmp'

    # if the algorithm is bwa-short, we must build use bwa aln to get an sai
    # file before calling bwa samse on that sai file, so we need to know how
    # to run bwa aln. Therefore, we must ensure there's an entry containing
    # those parameters
    if params['algorithm'] == 'bwa-short':
        if 'aln_params' not in params:
            raise InvalidArgumentApplicationError("With bwa-short, need to "
                                                  "specify a key 'aln_params' "
                                                  "and its value, a dictionary"
                                                  " to pass to bwa aln, since"
                                                  " bwa aln is an intermediate"
                                                  " step when doing "
                                                  "bwa-short.")

    # we have this params dict, with "algorithm" and "temp_dir", etc which are
    # not for any of the subcommands, so make a new params dict that is the
    # same as the original minus these addendums
    subcommand_params = {}
    for k, v in params.iteritems():
        if k not in ('algorithm', 'temp_dir', 'aln_params'):
            subcommand_params[k] = v

    # build index from database_fasta
    # get a temporary file name that is not in use
    _, index_prefix = mkstemp(dir=params['temp_dir'], suffix='')

    create_bwa_index_from_fasta_file(database_fasta, {'-p': index_prefix})

    # if the algorithm is bwasw, things are pretty simple. Just instantiate
    # the proper controller and set the files
    if params['algorithm'] == 'bwasw':
        bwa = BWA_bwasw(params=subcommand_params)
        files = {'prefix': index_prefix, 'query_fasta': query}

    # if the algorithm is bwa-short, it's not so simple
    elif params['algorithm'] == 'bwa-short':
        # we have to call bwa_aln to get the sai file needed for samse
        # use the aln_params we ensured we had above
        bwa_aln = BWA_aln(params=params['aln_params'])
        aln_files = {'prefix': index_prefix, 'fastq_in': query}
        # get the path to the sai file
        sai_file_path = bwa_aln(aln_files)['output'].name

        # we will use that sai file to run samse
        bwa = BWA_samse(params=subcommand_params)
        files = {'prefix': index_prefix, 'sai_in': sai_file_path,
                 'fastq_in': query}

    # run which ever app controller we decided was correct on the files
    # we set up
    result = bwa(files)

    # they both return a SAM file, so return that
    return result['output']