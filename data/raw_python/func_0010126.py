def scrape(args):
    """Scrape webpage content."""
    try:
        base_dir = os.getcwd()
        if args['out'] is None:
            args['out'] = []

        # Detect whether to save to a single or multiple files
        detect_output_type(args)

        # Split query input into local files and URLs
        split_input(args)

        if args['urls']:
            # Add URL extensions and schemes and update query and URLs
            urls_with_exts = [utils.add_url_suffix(x) for x in args['urls']]
            args['query'] = [utils.add_protocol(x) if x in args['urls'] else x
                             for x in urls_with_exts]
            args['urls'] = [x for x in args['query'] if x not in args['files']]

        # Print error if attempting to convert local files to HTML
        if args['files'] and args['html']:
            sys.stderr.write('Cannot convert local files to HTML.\n')
            args['files'] = []

        # Instantiate web crawler if necessary
        crawler = None
        if args['crawl'] or args['crawl_all']:
            crawler = Crawler(args)

        if args['single']:
            return write_single_file(args, base_dir, crawler)
        elif args['multiple']:
            return write_multiple_files(args, base_dir, crawler)

    except (KeyboardInterrupt, Exception):
        if args['html']:
            try:
                os.chdir(base_dir)
            except OSError:
                pass
        else:
            utils.remove_part_files()
        raise