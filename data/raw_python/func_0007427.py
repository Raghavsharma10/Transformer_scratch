def console_main():
    """Process command-line arguments."""
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--titles', action='store_true',
                        help='input files have column titles')
    parser.add_argument(
        '-j', '--join', choices=['inner', 'outer'],
        help=('The kind of left join to perform.  Outer join outputs left-hand '
              'rows which have no right hand match, while inner join discards '
              'such rows. Default: %(default)s'))
    parser.add_argument('-m', '--minscore', type=float,
                        help='Minimum match score: %(default)s')
    parser.add_argument('-c', '--count', type=int,
                help='Max number of rows to match (0 for all): %(default)s')
    parser.add_argument('-w', '--warp', type=float,
            help='N-gram warp, higher helps short strings: %(default)s')
    parser.add_argument('left', nargs=1, help='First CSV file')
    parser.add_argument('leftcolumn', nargs=1, type=int, help='Column in first CSV file')
    parser.add_argument('right', nargs=1, help='Second CSV file')
    parser.add_argument('rightcolumn', nargs=1, type=int, help='Column in second CSV file')
    parser.add_argument('outfile', nargs=1, help='Output CSV file')
    parser.set_defaults(
        titles=False, join='outer', minscore=0.24, count=0, warp=1.0)
    args = parser.parse_args()
    for path in [args.left[0], args.right[0]]:
        if not os.path.isfile(path):
            parser.error('File "%s" does not exist.' % path)
    if not (0 <= args.minscore <= 1.0):
        parser.error("Minimum score must be between 0 and 1")
    if not args.count >= 0:
        parser.error("Maximum number of matches per row must be non-negative.")
    if args.count == 0:
        args.count = None # to return all results
    main(args.left[0], args.leftcolumn[0], args.right[0], args.rightcolumn[0],
         args.outfile[0], args.titles, args.join, args.minscore, args.count,
         args.warp)