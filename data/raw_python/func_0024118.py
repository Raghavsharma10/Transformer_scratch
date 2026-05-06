def handle_compressed_input(inputfq, file_type="fastq"):
    """Return handles from compressed files according to extension.

    Check for which fastq input is presented and open a handle accordingly
    Can read from compressed files (gz, bz2, bgz) or uncompressed
    Relies on file extensions to recognize compression
    """
    ut.check_existance(inputfq)
    if inputfq.endswith(('.gz', 'bgz')):
        import gzip
        logging.info("Nanoget: Decompressing gzipped {} {}".format(file_type, inputfq))
        return gzip.open(inputfq, 'rt')
    elif inputfq.endswith('.bz2'):
        import bz2
        logging.info("Nanoget: Decompressing bz2 compressed {} {}".format(file_type, inputfq))
        return bz2.open(inputfq, 'rt')
    elif inputfq.endswith(('.fastq', '.fq', 'fasta', '.fa', '.fas')):
        return open(inputfq, 'r')
    else:
        logging.error("INPUT ERROR: Unrecognized file extension {}".format(inputfq))
        sys.exit('INPUT ERROR:\nUnrecognized file extension in {}\n'
                 'Supported are gz, bz2, bgz, fastq, fq, fasta, fa and fas'.format(inputfq))