def main(args=None):
    """Extract Ensembl IDs and store in tab-delimited text file.

    Parameters
    ----------
    args: argparse.Namespace object, optional
        The argument values. If not specified, the values will be obtained by
        parsing the command line arguments using the `argparse` module.

    Returns
    -------
    int
        Exit code (0 if no error occurred).
    """

    if args is None:
        # parse command-line arguments
        parser = get_argument_parser()
        args = parser.parse_args()

    input_file = args.annotation_file
    output_file = args.output_file
    species = args.species
    chrom_pat = args.chromosome_pattern
    field_name = args.field_name
    log_file = args.log_file
    quiet = args.quiet
    verbose = args.verbose

    # configure logger
    log_stream = sys.stdout
    if output_file == '-':
        # if we print output to stdout, redirect log messages to stderr
        log_stream = sys.stderr

    logger = misc.get_logger(log_stream = log_stream, log_file = log_file,
            quiet = quiet, verbose = verbose)

    if chrom_pat is None:
        chrom_pat = re.compile(ensembl.SPECIES_CHROMPAT[species])
    else:
        chrom_pat = re.compile(chrom_pat)

    logger.info('Regular expression used for filtering chromosome names: "%s"',
            chrom_pat.pattern)

    # for statistics
    types = Counter()
    sources = Counter()

    # primary information
    genes = Counter()
    gene_chroms = dict()
    gene_ids = dict()

    # secondary information
    genes2 = Counter()
    polymorphic = set()

    # list of chromosomes
    chromosomes = set()
    excluded_chromosomes = set()

    transcripts = {}
    gene_id = None
    gene_name = None

    i = 0
    missing = 0
    logger.info('Parsing data...')
    if input_file == '-':
        input_file = None
    with misc.smart_open_read(input_file, mode = 'rb', try_gzip = True) as fh:
        #if i >= 500000: break
        reader = csv.reader(fh, dialect = 'excel-tab')
        for l in reader:

            i += 1
            #if i % int(1e5) == 0:
            #   print '\r%d...' %(i), ; sys.stdout.flush() # report progress

            if len(l) > 1 and l[2] == field_name:
                attr = parse_attributes(l[8])
                type_ = attr['gene_biotype']

                if type_ not in ['protein_coding', 'polymorphic_pseudogene']:
                    continue

                chrom = l[0]

                # test whether chromosome is valid
                m = chrom_pat.match(chrom)
                if m is None:
                    excluded_chromosomes.add(chrom)
                    continue

                chromosomes.add(m.group())

                source = l[1]
                gene_id = attr['gene_id']
                try:
                    gene_name = attr['gene_name']
                except KeyError as e:
                    missing += 1
                    continue

                if gene_id in genes:
                    if genes[gene_id] != gene_name:
                        raise ValueError('Ensembl ID "%s" ' %(gene_id) +
                                'associated with multiple gene symbols.')
                else:
                    genes[gene_id] = gene_name

    logger.info('Done! (Parsed %d lines.)', i)

    logger.info('Excluded %d chromosomes:', len(excluded_chromosomes))
    logger.info(', '.join(sorted(excluded_chromosomes)))

    n = len(genes)
    m = len(set(genes.values()))

    logger.info('No. of chromosomes: %d', len(chromosomes))
    logger.info('No. of genes IDs: %d', n)
    logger.info('No. of gene names: %d', m)

    with misc.smart_open_write(output_file) as ofh:
        writer = csv.writer(ofh, dialect = 'excel-tab',
                lineterminator = os.linesep, quoting = csv.QUOTE_NONE)
        for g in sorted(genes.keys()):
            writer.writerow([g, genes[g]])

    return 0