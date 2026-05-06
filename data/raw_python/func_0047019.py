def check_rna_file(rna_file):
    """Check if bedtools is available and if the given RNA-Seq bam file is valid. """
    try:
        subprocess.check_output(['bedtools', '--version'])
    except OSError:
        log.error('Could not find bedtools in PATH. bedtools is required '
                  'for generating RNA coverage plot.')
        raise
    # Is this a valid BAM file? i.e., can pysam read it?
    try:
        is_bam_valid(rna_file)
    except ValueError:
        log.error('The given RNASeq BAM file is not valid')
        raise