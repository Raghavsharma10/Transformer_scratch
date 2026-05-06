def join_paired_end_reads_seqprep(
        reads1_infile_path,
        reads2_infile_path,
        outfile_label='seqprep',
        max_overlap_ascii_q_score='J',
        min_overlap=None,  # typical default vs 15
        max_mismatch_good_frac=None,  # typical default is 0.02,
        min_frac_matching=None,  # typical default is 0.9,
        phred_64=False,
        params={},
        working_dir=tempfile.gettempdir(),
        SuppressStderr=True,
        SuppressStdout=True,
        HALT_EXEC=False):
    """ Runs SeqPrep parameters to assemble paired-end reads.
        -reads1_infile_path : reads1.fastq infile path
        -reads2_infile_path : reads2.fastq infile path
        -max_overlap_ascii_q_score : 'J' for Illumina 1.8+ phred+33,
                                    representing a score of 41. See:
                                    http://en.wikipedia.org/wiki/FASTQ_format
        -min_overlap : minimum overall base pair overlap to merge two reads
        -max_mismatch_good_frac : maximum fraction of good quality mismatching
                                  bases to overlap reads
        -min_frac_matching : minimum fraction of matching bases to overlap
                             reads
        -phred_64 : if input is in phred+64. Output will always be phred+33.
        -params : other optional SeqPrep parameters

         NOTE: SeqPrep always outputs gzipped files
    """

    abs_r1_path = os.path.abspath(reads1_infile_path)
    abs_r2_path = os.path.abspath(reads2_infile_path)

    infile_paths = [abs_r1_path, abs_r2_path]

    # check / make absolute infile paths
    for p in infile_paths:
        if not os.path.exists(p):
            raise IOError('Infile not found at: %s' % p)

    # set up controller
    seqprep_app = SeqPrep(params=params,
                          WorkingDir=working_dir,
                          SuppressStderr=SuppressStderr,
                          SuppressStdout=SuppressStdout,
                          HALT_EXEC=HALT_EXEC)

    # required by SeqPrep to assemble:
    seqprep_app.Parameters['-f'].on(abs_r1_path)
    seqprep_app.Parameters['-r'].on(abs_r2_path)

    if outfile_label is not None:
        seqprep_app.Parameters['-s'].on(outfile_label + '_assembled.fastq.gz')
        seqprep_app.Parameters[
            '-1'].on(outfile_label + '_unassembled_R1.fastq.gz')
        seqprep_app.Parameters[
            '-2'].on(outfile_label + '_unassembled_R2.fastq.gz')
    else:
        raise ValueError("Must set an outfile_label in order to set",
                         " the -s, -1, & -2 options!")

    if min_overlap is not None:
        if isinstance(min_overlap, int) and min_overlap > 0:
                seqprep_app.Parameters['-o'].on(min_overlap)
        else:
            raise ValueError("min_overlap must be an int >= 0!")

    if max_mismatch_good_frac is not None:
        if isinstance(max_mismatch_good_frac, float) and 0.0 < max_mismatch_good_frac <= 1.0:
            seqprep_app.Parameters['-m'].on(max_mismatch_good_frac)
        else:
            raise ValueError(
                "max_mismatch_good_frac must be a float between 0.0-1.0!")

    if min_frac_matching is not None:
        if isinstance(min_frac_matching, float) and 0.0 < min_frac_matching <= 1.0:
            seqprep_app.Parameters['-n'].on(min_frac_matching)
        else:
            raise ValueError(
                "min_frac_matching must be a float between 0.0-1.0!")

    if max_overlap_ascii_q_score is not None:
        if isinstance(max_overlap_ascii_q_score, str) \
                and len(max_overlap_ascii_q_score) == 1:
            seqprep_app.Parameters['-y'].on(max_overlap_ascii_q_score)
        else:
            raise ValueError("max_overlap_ascii_q_score must be a single",
                             " ASCII character string. e.g. \'J\'!")

   # if input is phred+64
    if phred_64 is True:
        seqprep_app.Parameters['-6'].on()

    # run assembler
    result = seqprep_app()

    # Store output file path data to dict
    path_dict = {}
    path_dict['Assembled'] = result['Assembled'].name
    path_dict['UnassembledReads1'] = result['UnassembledReads1'].name
    path_dict['UnassembledReads2'] = result['UnassembledReads2'].name

   # sanity check that files actually exist in path lcoations
    for path in path_dict.values():
        if not os.path.exists(path):
            raise IOError('Output file not found at: %s' % path)

    return path_dict