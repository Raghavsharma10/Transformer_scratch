def main(args=None):
    """Extract protein-coding genes and store in tab-delimited text file.

    Parameters
    ----------
    args: argparse.Namespace object, optional
        The argument values. If not specified, the values will be obtained by
        parsing the command line arguments using the `argparse` module.

    Returns
    -------
    int
        Exit code (0 if no error occurred).
 
    Raises
    ------
    SystemError
        If the version of the Python interpreter is not >= 2.7.
    """

    vinfo = sys.version_info
    if not vinfo >= (2, 7):
        raise SystemError('Python interpreter version >= 2.7 required, '
                          'found %d.%d instead.' %(vinfo.major, vinfo.minor))

    if args is None:
        # parse command-line arguments
        parser = get_argument_parser()
        args = parser.parse_args()

    input_file = args.annotation_file
    output_file = args.output_file
    # species = args.species
    chrom_pat = args.chromosome_pattern
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

    #if chrom_pat is None:
    #    chrom_pat = ensembl.SPECIES_CHROMPAT[species]

    if chrom_pat is not None:
        logger.info('Regular expression used for filtering chromosome names: '
                    '"%s"', chrom_pat)

    if input_file == '-':
        input_file = sys.stdin

    if output_file == '-':
        output_file = sys.stdout

    genes = ensembl.get_protein_coding_genes(
        input_file,
        chromosome_pattern=chrom_pat)
    genes.to_csv(output_file, sep='\t', index=False)

    return 0