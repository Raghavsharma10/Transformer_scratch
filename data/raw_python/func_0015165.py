def _remove_api_url_from_link(link):
    '''Remove the API URL from the link if it is there'''
    if link.startswith(_api_url()):
        link = link[len(_api_url()):]
    if link.startswith(_api_url(mirror=True)):
        link = link[len(_api_url(mirror=True)):]
    return link