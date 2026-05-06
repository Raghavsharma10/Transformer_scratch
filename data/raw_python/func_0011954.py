def main(argv=None):
    """Takes crash data via args and generates a Socorro signature

    """
    parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG)
    parser.add_argument(
        '-v', '--verbose', help='increase output verbosity', action='store_true'
    )
    parser.add_argument(
        '--format', help='specify output format: csv, text (default)'
    )
    parser.add_argument(
        '--different-only', dest='different', action='store_true',
        help='limit output to just the signatures that changed',
    )
    parser.add_argument(
        'crashids', metavar='crashid', nargs='*', help='crash id to generate signatures for'
    )

    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    if args.format == 'csv':
        outputter = CSVOutput
    else:
        outputter = TextOutput

    api_token = os.environ.get('SOCORRO_API_TOKEN', '')

    generator = SignatureGenerator()
    if args.crashids:
        crashids_iterable = args.crashids
    elif not sys.stdin.isatty():
        # If a script is piping to this script, then isatty() returns False. If
        # there is no script piping to this script, then isatty() returns True
        # and if we do list(sys.stdin), it'll block waiting for input.
        crashids_iterable = list(sys.stdin)
    else:
        crashids_iterable = []

    if not crashids_iterable:
        parser.print_help()
        return 0

    with outputter() as out:
        for crash_id in crashids_iterable:
            crash_id = crash_id.strip()

            resp = fetch('/RawCrash/', crash_id, api_token)
            if resp.status_code == 404:
                out.warning('%s: does not exist.' % crash_id)
                continue
            if resp.status_code == 429:
                out.warning('API rate limit reached. %s' % resp.content)
                # FIXME(willkg): Maybe there's something better we could do here. Like maybe wait a
                # few minutes.
                return 1
            if resp.status_code == 500:
                out.warning('HTTP 500: %s' % resp.content)
                continue

            raw_crash = resp.json()

            # If there's an error in the raw crash, then something is wrong--probably with the API
            # token. So print that out and exit.
            if 'error' in raw_crash:
                out.warning('Error fetching raw crash: %s' % raw_crash['error'])
                return 1

            resp = fetch('/ProcessedCrash/', crash_id, api_token)
            if resp.status_code == 404:
                out.warning('%s: does not have processed crash.' % crash_id)
                continue
            if resp.status_code == 429:
                out.warning('API rate limit reached. %s' % resp.content)
                # FIXME(willkg): Maybe there's something better we could do here. Like maybe wait a
                # few minutes.
                return 1
            if resp.status_code == 500:
                out.warning('HTTP 500: %s' % resp.content)
                continue

            processed_crash = resp.json()

            # If there's an error in the processed crash, then something is wrong--probably with the
            # API token. So print that out and exit.
            if 'error' in processed_crash:
                out.warning('Error fetching processed crash: %s' % processed_crash['error'])
                return 1

            old_signature = processed_crash['signature']
            crash_data = convert_to_crash_data(raw_crash, processed_crash)

            result = generator.generate(crash_data)

            if not args.different or old_signature != result.signature:
                out.data(crash_id, old_signature, result, args.verbose)