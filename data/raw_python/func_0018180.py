def main():
    """Command line driver."""
    import argparse

    parser = argparse.ArgumentParser(
        prog=__taskname__,
        description='Remove horizontal stripes from ACS WFC post-SM4 data.')
    parser.add_argument(
        'arg0', metavar='input', type=str, help='Input file')
    parser.add_argument(
        'arg1', metavar='suffix', type=str, help='Output suffix')
    parser.add_argument(
        'maxiter', nargs='?', type=int, default=15, help='Max #iterations')
    parser.add_argument(
        'sigrej', nargs='?', type=float, default=2.0, help='Rejection sigma')
    parser.add_argument(
        '--stat', type=str, default='pmode1', help='Background statistics')
    parser.add_argument(
        '--lower', nargs='?', type=float, default=None,
        help='Lower limit for "good" pixels.')
    parser.add_argument(
        '--upper', nargs='?', type=float, default=None,
        help='Upper limit for "good" pixels.')
    parser.add_argument(
        '--binwidth', type=float, default=0.1,
        help='Bin width for distribution sampling.')
    parser.add_argument(
        '--mask1', nargs=1, type=str, default=None,
        help='Mask image for [SCI,1]')
    parser.add_argument(
        '--mask2', nargs=1, type=str, default=None,
        help='Mask image for [SCI,2]')
    parser.add_argument(
        '--dqbits', nargs='?', type=str, default=None,
        help='DQ bits to be considered "good".')
    parser.add_argument(
        '--rpt_clean', type=int, default=0,
        help='Number of *repeated* bias de-stripes to perform.')
    parser.add_argument(
        '--atol', nargs='?', type=float, default=0.01,
        help='Absolute tolerance to stop *repeated* bias de-stripes.')
    parser.add_argument(
        '-c', '--clobber', action="store_true", help='Clobber output')
    parser.add_argument(
        '-q', '--quiet', action='store_true',
        help='Do not print informational messages')
    parser.add_argument(
        '--version', action="version",
        version='{0} v{1} ({2})'.format(__taskname__, __version__, __vdate__))
    args = parser.parse_args()

    if args.mask1:
        mask1 = args.mask1[0]
    else:
        mask1 = args.mask1

    if args.mask2:
        mask2 = args.mask2[0]
    else:
        mask2 = args.mask2

    clean(args.arg0, args.arg1, stat=args.stat, maxiter=args.maxiter,
          sigrej=args.sigrej, lower=args.lower, upper=args.upper,
          binwidth=args.binwidth, mask1=mask1, mask2=mask2, dqbits=args.dqbits,
          rpt_clean=args.rpt_clean, atol=args.atol,
          clobber=args.clobber, verbose=not args.quiet)