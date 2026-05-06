def check_required_arguments(ribo_file, transcriptome_fasta, transcript_name=None):
    """Check required arguments of both riboplot and ribocount."""
    # Is this a valid BAM file? i.e., can pysam read it?
    try:
        is_bam_valid(ribo_file)
    except ValueError:
        log.error('The given RiboSeq BAM file is not valid')
        raise

    # Does the BAM file have an index? If not, create it.
    if not bam_has_index(ribo_file):
        log.info('Creating an index for the BAM file...')
        create_bam_index(ribo_file)

        if not bam_has_index(ribo_file):
            msg = ('Could not create an index for this BAM file. Is this a valid BAM file '
                   'and/or is the BAM file sorted by chromosomal coordinates?')
            log.error(msg)
            raise BamFileError(msg)

    # Is FASTA file valid?
    fasta_valid = False
    try:
        fasta_valid = is_fasta_valid(transcriptome_fasta)
    except IOError:
        log.error('Transcriptome FASTA file is not valid')
        raise

    if fasta_valid:
        if transcript_name:
            try:
                get_fasta_records(transcriptome_fasta, [transcript_name])
            except IOError:
                log.error('Could not get FASTA sequence of "{}" from transcriptome FASTA file'.format(transcript_name))
                raise
        else:
            # ribocount doesn't have a transcript option so we get the first
            # sequence name from the fasta file
            transcript_name = get_first_transcript_name(transcriptome_fasta)

        # check if transcript also exists in BAM
        with pysam.AlignmentFile(ribo_file, 'rb') as bam_file:
            if transcript_name not in bam_file.references:
                msg = 'Transcript "{}" does not exist in BAM file'.format(transcript_name)
                log.error(msg)
                raise ArgumentError(msg)