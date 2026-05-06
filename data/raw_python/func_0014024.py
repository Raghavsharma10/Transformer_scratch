def buildParser():
    ''' Builds the parser for reading the command line arguments'''
    parser = argparse.ArgumentParser(description='Bagfile reader')
    parser.add_argument('-b', '--bag', help='Bag file to read',
                        required=True, type=str)
    parser.add_argument('-s', '--series',
                        help='Msg data fields to graph',
                        required=True, nargs='*')
    parser.add_argument('-y ', '--ylim',
                        help='Set min and max y lim',
                        required=False, nargs=2)
    parser.add_argument('-c', '--combined',
                        help="Graph them all on one",
                        required=False, action="store_true", dest="sharey")
    return parser