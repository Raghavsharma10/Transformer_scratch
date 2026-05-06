def parse(url):
    """Parses a database URL."""

    config = {}

    url = urlparse.urlparse(url)

    # Remove query strings.
    path = url.path[1:]
    path = path.split('?', 2)[0]

    # Update with environment configuration.
    config.update({
        "DB": int(path or 0),
        "PASSWORD": url.password or None,
        "HOST": url.hostname or "localhost",
        "PORT": int(url.port or 6379),
    })

    return config