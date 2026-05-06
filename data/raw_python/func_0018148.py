def main():
    """Command line driver."""
    import argparse

    # Parse input parameters
    parser = argparse.ArgumentParser(
        prog=__taskname__,
        description=(
            'Run CALACS and standalone acs_destripe script on given post-SM4 '
            'ACS/WFC RAW full-frame or supported subarray image.'))
    parser.add_argument(
        'arg0', metavar='input', type=str, help='Input file')
    parser.add_argument(
        '--suffix', type=str, default='strp',
        help='Output suffix for acs_destripe')
    parser.add_argument(
        '--stat', type=str, default='pmode1', help='Background statistics')
    parser.add_argument(
        '--maxiter', type=int, default=15, help='Max #iterations')
    parser.add_argument(
        '--sigrej', type=float, default=2.0, help='Rejection sigma')
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
        '--sci1_mask', nargs=1, type=str, default=None,
        help='Mask image for calibrated [SCI,1]')
    parser.add_argument(
        '--sci2_mask', nargs=1, type=str, default=None,
        help='Mask image for calibrated [SCI,2]')
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
        '--nocte', action='store_true', help='Turn off CTE correction.')
    parser.add_argument(
        '--clobber', action='store_true', help='Clobber output')
    parser.add_argument(
        '-q', '--quiet', action="store_true",
        help='Do not print informational messages')
    parser.add_argument(
        '--version', action='version',
        version='{0} v{1} ({2})'.format(__taskname__, __version__, __vdate__))
    options = parser.parse_args()

    if options.sci1_mask:
        mask1 = options.sci1_mask[0]
    else:
        mask1 = options.sci1_mask

    if options.sci2_mask:
        mask2 = options.sci2_mask[0]
    else:
        mask2 = options.sci2_mask

    destripe_plus(options.arg0, suffix=options.suffix,
                  maxiter=options.maxiter, sigrej=options.sigrej,
                  lower=options.lower, upper=options.upper,
                  binwidth=options.binwidth,
                  scimask1=mask1, scimask2=mask2, dqbits=options.dqbits,
                  rpt_clean=options.rpt_clean, atol=options.atol,
                  cte_correct=not options.nocte, clobber=options.clobber,
                  verbose=not options.quiet)