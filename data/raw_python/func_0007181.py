def download_file_insecure(url, target):
    '''
    Use Python to download the file, even though it cannot authenticate the
    connection.
    '''
    try:
        from urllib.request import urlopen
    except ImportError:
        from urllib2 import urlopen
    src = dst = None
    try:
        src = urlopen(url)
        # Read/write all in one block, so we don't create a corrupt file
        # if the download is interrupted.
        data = src.read()
        dst = open(target, 'wb')
        dst.write(data)
    finally:
        if src:
            src.close()
        if dst:
            dst.close()