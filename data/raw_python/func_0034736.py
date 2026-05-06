def _load_github_hooks(github_url='https://api.github.com'):
    """Request GitHub's IP block from their API.

    Return the IP network.

    If we detect a rate-limit error, raise an error message stating when
    the rate limit will reset.

    If something else goes wrong, raise a generic 503.
    """
    try:
        resp = requests.get(github_url + '/meta')
        if resp.status_code == 200:
            return resp.json()['hooks']
        else:
            if resp.headers.get('X-RateLimit-Remaining') == '0':
                reset_ts = int(resp.headers['X-RateLimit-Reset'])
                reset_string = time.strftime('%a, %d %b %Y %H:%M:%S GMT',
                                             time.gmtime(reset_ts))
                raise ServiceUnavailable('Rate limited from GitHub until ' +
                                         reset_string)
            else:
                raise ServiceUnavailable('Error reaching GitHub')
    except (KeyError, ValueError, requests.exceptions.ConnectionError):
        raise ServiceUnavailable('Error reaching GitHub')