def download_extract(url):
    """download and extract file."""
    logger.info("Downloading %s", url)
    request = urllib2.Request(url)
    request.add_header('User-Agent',
                       'caelum/0.1 +https://github.com/nrcharles/caelum')
    opener = urllib2.build_opener()
    with tempfile.TemporaryFile(suffix='.zip', dir=env.WEATHER_DATA_PATH) \
            as local_file:
        logger.debug('Saving to temporary file %s', local_file.name)
        local_file.write(opener.open(request).read())
        compressed_file = zipfile.ZipFile(local_file, 'r')
        logger.debug('Extracting %s', compressed_file)
        compressed_file.extractall(env.WEATHER_DATA_PATH)
        local_file.close()