def download_url(result, directory, max_bytes):
    """
    Download a file.

    :param result: This field contains a url from which data will be
        downloaded.
    :param directory: This field is the target directory to which data will be
        downloaded.
    :param max_bytes: This field is the maximum file size in bytes.
    """
    response = requests.get(result['download_url'], stream=True)

    # TODO: make this more robust by parsing the URL
    filename = response.url.split('/')[-1]
    filename = re.sub(r'\?.*$', '', filename)
    filename = '{}-{}'.format(result['user']['id'], filename)

    size = int(response.headers['Content-Length'])

    if size > max_bytes:
        logging.info('Skipping {}, {} > {}'.format(filename, format_size(size),
                                                   format_size(max_bytes)))

        return

    logging.info('Downloading {} ({})'.format(filename, format_size(size)))

    output_path = os.path.join(directory, filename)

    try:
        stat = os.stat(output_path)

        if stat.st_size == size:
            logging.info('Skipping "{}"; exists and is the right size'.format(
                filename))

            return
        else:
            logging.info('Removing "{}"; exists and is the wrong size'.format(
                filename))

            os.remove(output_path)
    except OSError:
        # TODO: check errno here?
        pass

    with open(output_path, 'wb') as f:
        total_length = response.headers.get('content-length')
        total_length = int(total_length)
        dl = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                dl += len(chunk)
                f.write(chunk)
                d = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]%d%s" % ('.' * d,
                                                   '' * (50 - d),
                                                   d * 2,
                                                   '%'))
                sys.stdout.flush
        print("\n")

    logging.info('Downloaded {}'.format(filename))