def patch_match_hostname():
    """Fixes https://github.com/boto/boto/issues/2836"""


    _old_match_hostname = ssl.match_hostname

    def _new_match_hostname(cert, hostname):
        if hostname.endswith('.s3.amazonaws.com'):
            pos = hostname.find('.s3.amazonaws.com')
            hostname = hostname[:pos].replace('.', '') + hostname[pos:]
        return _old_match_hostname(cert, hostname)

    ssl.match_hostname = _new_match_hostname