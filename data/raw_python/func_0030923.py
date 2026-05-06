def main(args=None):
    """Script body."""

    if args is None:
        # parse command-line arguments 
        parser = get_argument_parser()
        args = parser.parse_args()

    fasta_file = args.fasta_file
    species = args.species
    chrom_pat = args.chromosome_pattern
    output_file = args.output_file
    
    log_file = args.log_file
    quiet = args.quiet
    verbose = args.verbose

    # configure root logger
    log_stream = sys.stdout
    if output_file == '-':
        # if we print output to stdout, redirect log messages to stderr
        log_stream = sys.stderr

    logger = misc.get_logger(log_stream=log_stream, log_file=log_file,
                             quiet=quiet, verbose=verbose)

    # generate regular expression object from the chromosome pattern
    if chrom_pat is None:
        chrom_pat = ensembl.SPECIES_CHROMPAT[species]
    chrom_re = re.compile(chrom_pat)

    # filter the FASTA file
    # note: each chromosome sequence is temporarily read into memory,
    # so this script has a large memory footprint
    with \
        misc.smart_open_read(
            fasta_file, mode='r', encoding='ascii', try_gzip=True
        ) as fh, \
        misc.smart_open_write(
            output_file, mode='w', encoding='ascii'
        ) as ofh:

        # inside = False
        reader = FastaReader(fh)
        for seq in reader:
            chrom = seq.name.split(' ', 1)[0]
            if chrom_re.match(chrom) is None:
                logger.info('Ignoring chromosome "%s"...', chrom)
                continue
            seq.name = chrom
            seq.append_fasta(ofh)

    return 0