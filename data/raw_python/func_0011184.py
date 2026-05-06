def should_bypass_proxies(url):
    """
    Returns whether we should bypass proxies or not.
    """
    get_proxy = lambda k: os.environ.get(k) or os.environ.get(k.upper())

    # First check whether no_proxy is defined. If it is, check that the URL
    # we're getting isn't in the no_proxy list.
    no_proxy = get_proxy('no_proxy')
    netloc = urlparse(url).netloc

    if no_proxy:
        # We need to check whether we match here. We need to see if we match
        # the end of the netloc, both with and without the port.
        no_proxy = no_proxy.replace(' ', '').split(',')

        ip = netloc.split(':')[0]
        if is_ipv4_address(ip):
            for proxy_ip in no_proxy:
                if is_valid_cidr(proxy_ip):
                    if address_in_network(ip, proxy_ip):
                        return True
        else:
            for host in no_proxy:
                if netloc.endswith(host) or netloc.split(':')[0].endswith(host):
                    # The URL does match something in no_proxy, so we don't want
                    # to apply the proxies on this URL.
                    return True

    # If the system proxy settings indicate that this URL should be bypassed,
    # don't proxy.
    # The proxy_bypass function is incredibly buggy on OS X in early versions
    # of Python 2.6, so allow this call to fail. Only catch the specific
    # exceptions we've seen, though: this call failing in other ways can reveal
    # legitimate problems.
    try:
        bypass = proxy_bypass(netloc)
    except (TypeError, socket.gaierror):
        bypass = False

    if bypass:
        return True

    return False