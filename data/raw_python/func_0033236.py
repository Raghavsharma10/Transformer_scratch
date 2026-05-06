async def download(resource_url):
    '''
    Download given resource_url
    '''
    scheme = resource_url.parsed.scheme
    if scheme in ('http', 'https'):
        await download_http(resource_url)
    elif scheme in ('git', 'git+https', 'git+http'):
        await download_git(resource_url)
    else:
        raise ValueError('Unknown URL scheme: "%s"' % scheme)