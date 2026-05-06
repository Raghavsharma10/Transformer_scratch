def get_tile(url):
    """
    Threadable function to retrieve map tiles from the internet
    :param url: URL of tile to get
    """
    log = ""
    connection = None

    try:
        if six.PY3:
            connection = urlopen(url=url, timeout=2)  # NOQA
        else:
            connection = urlopen(url=url)
    except Exception as e:
        log += "MTST: ERROR Tilegetter error: %s " % str([type(e), e, url])

    content = ""

    # Read and return requested content
    if connection:
        try:
            content = connection.read()
        except (socket.timeout, socket.error) as e:
            log += "MTST: ERROR Tilegetter error: %s " % str([type(e), e])

        connection.close()
    else:
        log += "MTST: ERROR Got no connection."

    return content, log