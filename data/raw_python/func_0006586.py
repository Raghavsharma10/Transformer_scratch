def main(argv=None):
    '''
    Main entry-point for calling layouts directly as a program.
    '''
    # Prep argparse
    ap = argparse.ArgumentParser(
        description='Basic query options for Python HID-IO Layouts repository',
    )
    ap.add_argument('--list', action='store_true', help='List available layout aliases.')
    ap.add_argument('--get', metavar='NAME', help='Retrieve the given layout, and return the JSON data')

    # Parse arguments
    args = ap.parse_args(argv)

    # Create layouts context manager
    mgr = Layouts()

    # Check if generating a list
    if args.list:
        for name in mgr.list_layouts():
            print(name)

    # Retrieve JSON layout
    if args.get is not None:
        layout = mgr.get_layout(args.get)
        print(json.dumps(layout.json()))