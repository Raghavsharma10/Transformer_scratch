def join_paired_end_reads_fastqjoin(
        reads1_infile_path,
        reads2_infile_path,
        perc_max_diff=None,  # typical default is 8
        min_overlap=None,  # typical default is 6
        outfile_label='fastqjoin',
        params={},
        working_dir=tempfile.gettempdir(),
        SuppressStderr=True,
        SuppressStdout=True,
        HALT_EXEC=False):
    """ Runs fastq-join, with default parameters to assemble paired-end reads.
        Returns file path string.

        -reads1_infile_path : reads1.fastq infile path
        -reads2_infile_path : reads2.fastq infile path
        -perc_max_diff : maximum % diff of overlap differences allowed
        -min_overlap : minimum allowed overlap required to assemble reads
        -outfile_label : base name for output files.
        -params : dictionary of application controller parameters

    """
    abs_r1_path = os.path.abspath(reads1_infile_path)
    abs_r2_path = os.path.abspath(reads2_infile_path)

    infile_paths = [abs_r1_path, abs_r2_path]

    # check / make absolute infile paths
    for p in infile_paths:
        if not os.path.exists(p):
            raise IOError('File not found at: %s' % p)

    fastq_join_app = FastqJoin(params=params,
                               WorkingDir=working_dir,
                               SuppressStderr=SuppressStderr,
                               SuppressStdout=SuppressStdout,
                               HALT_EXEC=HALT_EXEC)

    # set param. Helps with QIIME integration to have these values
    # set to None by default. This way we do not have to worry
    # about changes in default behaviour of the wrapped
    # application
    if perc_max_diff is not None:
        if isinstance(perc_max_diff, int) and 0 <= perc_max_diff <= 100:
            fastq_join_app.Parameters['-p'].on(perc_max_diff)
        else:
            raise ValueError("perc_max_diff must be int between 0-100!")

    if min_overlap is not None:
        if isinstance(min_overlap, int) and 0 < min_overlap:
            fastq_join_app.Parameters['-m'].on(min_overlap)
        else:
            raise ValueError("min_overlap must be an int >= 0!")

    if outfile_label is not None:
        if isinstance(outfile_label, str):
            fastq_join_app.Parameters['-o'].on(outfile_label + '.')
        else:
            raise ValueError("outfile_label must be a string!")
    else:
        pass

    # run assembler
    result = fastq_join_app(infile_paths)

    # Store output file path data to dict
    path_dict = {}
    path_dict['Assembled'] = result['Assembled'].name
    path_dict['UnassembledReads1'] = result['UnassembledReads1'].name
    path_dict['UnassembledReads2'] = result['UnassembledReads2'].name

    # sanity check that files actually exist in path lcoations
    for path in path_dict.values():
        if not os.path.exists(path):
            raise IOError('Output file not found at: %s' % path)

    # fastq-join automatically appends: 'join', 'un1', or 'un2'
    # to the end of the file names. But we want to rename them so
    # they end in '.fastq'. So, we iterate through path_dict to
    # rename the files and overwrite the dict values.
    for key, file_path in path_dict.items():
        new_file_path = file_path + '.fastq'
        shutil.move(file_path, new_file_path)
        path_dict[key] = new_file_path

    return path_dict