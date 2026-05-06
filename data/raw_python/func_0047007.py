def open_pysam_file(fname, ftype):
    """Open a BAM or FASTA file with pysam (for use with "with" statement)"""
    try:
        if ftype == 'bam':
            fpysam = pysam.AlignmentFile(fname, 'rb')
        elif ftype == 'fasta':
            fpysam = pysam.FastaFile(fname)
        yield fpysam
    except:
        raise
    else:
        fpysam.close()