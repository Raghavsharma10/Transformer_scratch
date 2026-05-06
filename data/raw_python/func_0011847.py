def main():
    """Takes crash data via stdin and generates a Socorro signature"""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        '-v', '--verbose', help='increase output verbosity', action='store_true'
    )
    args = parser.parse_args()

    generator = SignatureGenerator(debug=args.verbose)

    crash_data = json.loads(sys.stdin.read())

    ret = generator.generate(crash_data)

    print(json.dumps(ret, indent=2))