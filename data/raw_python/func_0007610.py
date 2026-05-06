def download(source=None, username=None, directory='.', max_size='128m',
             quiet=None, debug=None):
    """
    Download public data from Open Humans.

    :param source: This field is the data source from which to download. It's
        default value is None.
    :param username: This fiels is username of user. It's default value is
        None.
    :param directory: This field is the target directory to which data is
        downloaded.
    :param max_size: This field is the maximum file size. It's default value is
        128m.
    :param quiet: This field is the logging level. It's default value is
        None.
    :param debug: This field is the logging level. It's default value is
        None.
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    elif quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.debug("Running with source: '{}'".format(source) +
                  " and username: '{}'".format(username) +
                  " and directory: '{}'".format(directory) +
                  " and max-size: '{}'".format(max_size))

    signal.signal(signal.SIGINT, signal_handler_cb)

    max_bytes = parse_size(max_size)

    options = {}

    if source:
        options['source'] = source

    if username:
        options['username'] = username

    page = '{}?{}'.format(BASE_URL_API, urlencode(options))

    results = []
    counter = 1

    logging.info('Retrieving metadata')

    while True:
        logging.info('Retrieving page {}'.format(counter))

        response = get_page(page)
        results = results + response['results']

        if response['next']:
            page = response['next']
        else:
            break

        counter += 1

    logging.info('Downloading {} files'.format(len(results)))

    download_url_partial = partial(download_url, directory=directory,
                                   max_bytes=max_bytes)

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for value in executor.map(download_url_partial, results):
            if value:
                logging.info(value)