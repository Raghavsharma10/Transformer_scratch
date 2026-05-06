def K2onSilicon_main(args=None):
    """Function called when `K2onSilicon` is executed on the command line."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Run K2onSilicon to find which targets in a "
                    "list call on active silicon for a given K2 campaign.")
    parser.add_argument('csv_file', type=str,
                        help="Name of input csv file with targets, column are "
                             "Ra_degrees, Dec_degrees, Kepmag")
    parser.add_argument('campaign', type=int, help='K2 Campaign number')
    args = parser.parse_args(args)
    K2onSilicon(args.csv_file, args.campaign)