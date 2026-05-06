def extract_from_bam(params):
    """Extracts metrics from bam.

    Worker function per chromosome
    loop over a bam file and create list with tuples containing metrics:
    -qualities
    -aligned qualities
    -lengths
    -aligned lengths
    -mapping qualities
    -edit distances to the reference genome scaled by read length
    """
    bam, chromosome = params
    samfile = pysam.AlignmentFile(bam, "rb")
    return [
        (read.query_name,
         nanomath.ave_qual(read.query_qualities),
         nanomath.ave_qual(read.query_alignment_qualities),
         read.query_length,
         read.query_alignment_length,
         read.mapping_quality,
         get_pID(read))
        for read in samfile.fetch(reference=chromosome, multiple_iterators=True)
        if not read.is_secondary]