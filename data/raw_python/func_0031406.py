def main(args=None):
    """Extract all exon annotations of protein-coding genes."""

    if args is None:
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

    # configure root logger
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

    chromosomes = set()
    excluded_chromosomes = set()
    i = 0
    exons = 0
    logger.info('Parsing data...')
    if input_file == '-':
        input_file = None
    with misc.smart_open_read(input_file, mode = 'rb', try_gzip = True) as fh, \
            misc.smart_open_write(output_file) as ofh:
        #if i >= 500000: break
        reader = csv.reader(fh, dialect = 'excel-tab')
        writer = csv.writer(ofh, dialect = 'excel-tab', lineterminator = os.linesep,
                quoting = csv.QUOTE_NONE , quotechar = '|')
        for l in reader:
            i += 1
            #if i % int(1e5) == 0:
            #   print '\r%d...' %(i), ; sys.stdout.flush() # report progress
            if len(l) > 1 and l[2] == field_name:
                attr = parse_attributes(l[8])
                type_ = attr['gene_biotype']
                if type_ in ['protein_coding','polymorphic_pseudogene']:

                    # test whether chromosome is valid
                    chrom = l[0]
                    m = chrom_pat.match(chrom)
                    if m is None:
                        excluded_chromosomes.add(chrom)
                        continue

                    chromosomes.add(chrom)
                    writer.writerow(l)
                    exons += 1

    logger.info('Done! (Parsed %d lines.)', i)

    logger.info('')
    logger.info('Gene chromosomes (%d):', len(chromosomes))
    logger.info('\t' + ', '.join(sorted(chromosomes)))
    logger.info('')
    logger.info('Excluded chromosomes (%d):', len(excluded_chromosomes))
    logger.info('\t' + ', '.join(sorted(excluded_chromosomes)))
    logger.info('')
    logger.info('Total no. of exons: %d' %(exons))

    return 0