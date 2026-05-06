def download(url, filename):
    """download and extract file."""
    logger.info("Downloading %s", url)
    request = urllib2.Request(url)
    request.add_header('User-Agent',
                       'caelum/0.1 +https://github.com/nrcharles/caelum')
    opener = urllib2.build_opener()
    local_file = open(filename, 'w')
    local_file.write(opener.open(request).read())
    local_file.close()