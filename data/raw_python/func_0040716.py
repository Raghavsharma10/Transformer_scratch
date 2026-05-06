def main(args_list=None):
    """
    Wrapper for find_schemes if called from command line
    """
    args_list = args_list or sys.argv[1:]
    parser = argparse.ArgumentParser(description='Discover schemes of given stanza file')
    parser.add_argument(
        'infile',
        type=argparse.FileType('r'),
    )
    parser.add_argument(
        'outfile',
        help='Where the result is written to. Default: stdout',
        nargs='?',
        type=argparse.FileType('w'),
        default=sys.stdout,
    )
    parser.add_argument(
        '-t --init-type',
        help='Whether to initialize theta uniformly (u), with the orthographic similarity '
             'measure (o), or using CELEX pronunciations and definition of rhyme (p). '
             'The last one requires you to have CELEX on your machine.',
        dest='init_type',
        choices=('u', 'o', 'p', 'd'),
        default='o',
    )
    parser.add_argument(
        '-i, --iterations',
        help='Number of iterations (default: 10)',
        dest='num_iterations',
        type=int,
        default=10,
    )
    parser.add_argument(
        '-v', '--verbose',
        help="Verbose output",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    args = parser.parse_args(args_list)
    logging.basicConfig(level=args.loglevel)

    stanzas = load_stanzas(args.infile)

    init_function = None
    if args.init_type == 'u':
        init_function = init_uniform_ttable
    elif args.init_type == 'o':
        init_function = init_basicortho_ttable
    elif args.init_type == 'p':
        init_function = celex.init_perfect_ttable
    elif args.init_type == 'd':
        init_function = init_difflib_ttable

    results = find_schemes(stanzas, init_function, args.num_iterations)

    print_results(results, args.outfile)