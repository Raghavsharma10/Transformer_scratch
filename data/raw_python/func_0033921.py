def parse(url):
    """Parses a search URL."""

    config = {}

    url = urlparse.urlparse(url)

    # Remove query strings.
    path = url.path[1:]
    path = path.split('?', 2)[0]

    if url.scheme in SCHEMES:
        config["ENGINE"] = SCHEMES[url.scheme]

    if url.scheme in USES_URL:
        config["URL"] = urlparse.urlunparse(("http",) + url[1:])

    if url.scheme in USES_INDEX:
        if path.endswith("/"):
            path = path[:-1]

        split = path.rsplit("/", 1)

        if len(split) > 1:
            path = split[:-1]
            index = split[-1]
        else:
            path = ""
            index = split[0]

        config.update({
            "URL": urlparse.urlunparse(("http",) + url[1:2] + (path,) + url[3:]),
            "INDEX_NAME": index,
        })

    if url.scheme in USES_PATH:
        config.update({
            "PATH": path,
        })

    return config