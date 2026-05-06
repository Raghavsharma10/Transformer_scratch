def check_bam(bam, samtype="bam"):
    """Check if bam file is valid.

    Bam file should:
    - exists
    - has an index (create if necessary)
    - is sorted by coordinate
    - has at least one mapped read
    """
    ut.check_existance(bam)
    samfile = pysam.AlignmentFile(bam, "rb")
    if not samfile.has_index():
        pysam.index(bam)
        samfile = pysam.AlignmentFile(bam, "rb")  # Need to reload the samfile after creating index
        logging.info("Nanoget: No index for bam file could be found, created index.")
    if not samfile.header['HD']['SO'] == 'coordinate':
        logging.error("Nanoget: Bam file {} not sorted by coordinate!.".format(bam))
        sys.exit("Please use a bam file sorted by coordinate.")
    if samtype == "bam":
        logging.info("Nanoget: Bam file {} contains {} mapped and {} unmapped reads.".format(
            bam, samfile.mapped, samfile.unmapped))
        if samfile.mapped == 0:
            logging.error("Nanoget: Bam file {} does not contain aligned reads.".format(bam))
            sys.exit("FATAL: not a single read was mapped in bam file {}".format(bam))
    return samfile