def bam_has_index(bam_file):
    """Check if bam file has an index. Returns True/False."""
    has_index = None
    with pysam.AlignmentFile(bam_file, 'rb') as bam_fileobj:
        try:
            bam_fileobj.fetch(bam_fileobj.references[0])
        except ValueError as err:
            if err.message == 'fetch called on bamfile without index':
                bam_fileobj.close()
                has_index = False
        else:
            has_index = True
    return has_index