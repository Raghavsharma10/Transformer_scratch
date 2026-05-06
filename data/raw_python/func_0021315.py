def parse_args():
    """Parse arguments from the command line"""

    parser = argparse.ArgumentParser(description=TO_KIBANA5_DESC_MSG)

    parser.add_argument('-s', '--source', dest='src_path', \
        required=True, help='source directory')
    parser.add_argument('-d', '--dest', dest='dest_path', \
        required=True, help='destination directory')

    parser.add_argument('-o', '--old-size', dest='old_size', \
        default='0', help='aggregation old size')
    parser.add_argument('-n', '--new-size', dest='new_size', \
        default='1000', help='aggregation new size')

    parser.add_argument('-g', '--debug', dest='debug',
                        action='store_true')

    return parser.parse_args()