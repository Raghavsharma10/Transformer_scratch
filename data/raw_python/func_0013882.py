def url_to_host(url):
    """convert a url to a host (ip or domain)

    :param url: url string
    :returns: host: domain name or ipv4/v6 address
    :rtype: str
    :raises: ValueError: given an illegal url that without a ip or domain name
    """

    regex_url = r"([a-z][a-z0-9+\-.]*://)?" + \
                r"([a-z0-9\-._~%!$&'()*+,;=]+@)?" + \
                r"([a-z0-9\-._~%]+" + \
                r"|\[[a-z0-9\-._~%!$&'()*+,;=:]+\])?" + \
                r"(:(?P<port>[0-9]+))?"

    m = re.match(regex_url, url, re.IGNORECASE)
    if m and m.group(3):
        return url[m.start(3): m.end(3)]
    else:
        raise ValueError("URL without a valid host or ip")