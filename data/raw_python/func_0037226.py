def from_url(location):
    """ HTTP request for page at location returned as string

    malformed url returns ValueError
    nonexistant IP returns URLError
    wrong subnet IP return URLError
    reachable IP, no HTTP server returns URLError
    reachable IP, HTTP, wrong page returns HTTPError
    """
    req = urllib.request.Request(location)
    with urllib.request.urlopen(req) as response:
        the_page = response.read().decode()
        return the_page