def command_line_runner():
    """Handle command-line interaction."""
    parser = get_parser()
    args = vars(parser.parse_args())
    if args['version']:
        print(__version__)
        return
    if args['clear_cache']:
        utils.clear_cache()
        print('Cleared {0}.'.format(utils.CACHE_DIR))
        return
    if not args['query']:
        parser.print_help()
        return

    # Enable cache unless user sets environ variable SCRAPE_DISABLE_CACHE
    if not os.getenv('SCRAPE_DISABLE_CACHE'):
        utils.enable_cache()

    # Save images unless user sets environ variable SCRAPE_DISABLE_IMGS
    if os.getenv('SCRAPE_DISABLE_IMGS'):
        args['no_images'] = True

    # Prompt user for filetype if none specified
    prompt_filetype(args)

    # Prompt user to save images when crawling (for pdf and HTML formats)
    prompt_save_images(args)

    # Scrape webpage content
    scrape(args)