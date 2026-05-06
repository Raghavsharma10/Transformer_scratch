def get_netrc_auth(url):
    """Returns the Requests tuple auth for a given url from netrc."""

    try:
        from netrc import netrc, NetrcParseError

        netrc_path = None

        for f in NETRC_FILES:
            try:
                loc = os.path.expanduser('~/{0}'.format(f))
            except KeyError:
                # os.path.expanduser can fail when $HOME is undefined and
                # getpwuid fails. See http://bugs.python.org/issue20164 &
                # https://github.com/kennethreitz/requests/issues/1846
                return

            if os.path.exists(loc):
                netrc_path = loc
                break

        # Abort early if there isn't one.
        if netrc_path is None:
            return

        ri = urlparse(url)

        # Strip port numbers from netloc
        host = ri.netloc.split(':')[0]

        try:
            _netrc = netrc(netrc_path).authenticators(host)
            if _netrc:
                # Return with login / password
                login_i = (0 if _netrc[0] else 1)
                return (_netrc[login_i], _netrc[2])
        except (NetrcParseError, IOError):
            # If there was a parsing error or a permissions issue reading the file,
            # we'll just skip netrc auth
            pass

    # AppEngine hackiness.
    except (ImportError, AttributeError):
        pass