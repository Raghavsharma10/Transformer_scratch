def main():
    """Takes a crash id, pulls down data from Socorro, generates signature data"""
    parser = argparse.ArgumentParser(
        formatter_class=WrappedTextHelpFormatter,
        description=DESCRIPTION
    )
    parser.add_argument(
        '-v', '--verbose', help='increase output verbosity', action='store_true'
    )
    parser.add_argument(
        'crashid', help='crash id to generate signatures for'
    )

    args = parser.parse_args()

    api_token = os.environ.get('SOCORRO_API_TOKEN', '')

    crash_id = args.crashid.strip()

    resp = fetch('/RawCrash/', crash_id, api_token)
    if resp.status_code == 404:
        printerr('%s: does not exist.' % crash_id)
        return 1
    if resp.status_code == 429:
        printerr('API rate limit reached. %s' % resp.content)
        # FIXME(willkg): Maybe there's something better we could do here. Like maybe wait a
        # few minutes.
        return 1
    if resp.status_code == 500:
        printerr('HTTP 500: %s' % resp.content)
        return 1

    raw_crash = resp.json()

    # If there's an error in the raw crash, then something is wrong--probably with the API
    # token. So print that out and exit.
    if 'error' in raw_crash:
        print('Error fetching raw crash: %s' % raw_crash['error'], file=sys.stderr)
        return 1

    resp = fetch('/ProcessedCrash/', crash_id, api_token)
    if resp.status_code == 404:
        printerr('%s: does not have processed crash.' % crash_id)
        return 1
    if resp.status_code == 429:
        printerr('API rate limit reached. %s' % resp.content)
        # FIXME(willkg): Maybe there's something better we could do here. Like maybe wait a
        # few minutes.
        return 1
    if resp.status_code == 500:
        printerr('HTTP 500: %s' % resp.content)
        return 1

    processed_crash = resp.json()

    # If there's an error in the processed crash, then something is wrong--probably with the
    # API token. So print that out and exit.
    if 'error' in processed_crash:
        printerr('Error fetching processed crash: %s' % processed_crash['error'])
        return 1

    crash_data = convert_to_crash_data(raw_crash, processed_crash)

    print(json.dumps(crash_data, indent=2))