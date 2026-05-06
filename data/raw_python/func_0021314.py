def main():
    """Read a directory containing json files for Kibana panels,
    beautify them and replace size value in aggregations as specified
    through corresponding params params.
    """
    args = parse_args()
    configure_logging(args.debug)

    src_path = args.src_path
    dest_path = args.dest_path
    old_str1 = '\\"size\\":' + args.old_size
    old_str2 = '\\"size\\": ' + args.old_size
    new_str = '\\"size\\":' + args.new_size

    logging.info('Input path: %s', src_path)
    logging.info('Output path: %s', dest_path)
    logging.info('old str: %s', old_str1)
    logging.info('old str: %s', old_str2)
    logging.info('new str: %s', new_str)

    if os.path.abspath(src_path) == os.path.abspath(dest_path):
        logging.error('source and destination directiories must be different')
        sys.exit(1)

    # Iterate over input files
    json_files = [f for f in os.listdir(src_path) if f.endswith('.json')]
    for filename in json_files:

        in_file_path = os.path.join(src_path, filename)
        in_file_path = os.path.join(src_path, filename)

        out_file_path = os.path.join(dest_path, filename)
        logging.info('INPUT FILE: %s',in_file_path)
        logging.info('OUTPUT FILE: %s',out_file_path)
        # First beautify input
        pretty = utils.beautify(filename=in_file_path)

        # Iterate the beautified json string line by line
        pretty_replaced = utils.replace(pretty, old_str1, new_str)
        pretty_replaced = utils.replace(pretty_replaced, old_str2, new_str)

        with open(out_file_path, 'w') as output_file:
            output_file.write(pretty_replaced)

    logging.info('This is the end.')