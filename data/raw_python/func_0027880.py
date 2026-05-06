def entry():
    """Parse command line arguments and run utilities."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'action', help='Action to take',
        choices=['from_hex', 'to_rgb', 'to_hex'],
    )
    parser.add_argument(
        'value', help='Value for the action',
    )
    parsed = parser.parse_args()
    if parsed.action != "from_hex":
        try:
            parsed.value = int(parsed.value)
        except ValueError:
            raise argparse.ArgumentError(
                "Value for this action should be an integer",
            )
    print(globals()[parsed.action](parsed.value))