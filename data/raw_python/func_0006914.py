def main(argv: Sequence[str] = SYS_ARGV) -> int:
    """Execute CLI commands."""
    args = default_parser().parse_args(argv)

    try:
        seq = POPULATIONS[args.population]  # type: Sequence
    except KeyError:
        try:
            with open(args.population, 'r', encoding=args.encoding) as file_:
                seq = list(file_)
        except (OSError, UnicodeError) as ex:
            print(ex, file=sys.stderr)
            return 1

    main_key = key(seq=seq, nteeth=args.nteeth, delimiter=args.delimiter)
    print(main_key)

    if args.stats:
        print('*', len(main_key), 'characters')
        print('*', args.nteeth, 'samples from a population of', len(seq))
        print(
            '* entropy {sign} {nbits} bits'.format(
                sign='~' if args.delimiter else '<',
                nbits=round(math.log(len(seq), 2) * args.nteeth, 2),
            ),
        )

    return 0