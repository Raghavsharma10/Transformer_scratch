def data(link):
    '''Returns a dictionary from requested link'''
    link = _remove_api_url_from_link(link)
    req = _get_from_dapi_or_mirror(link)
    return _process_req(req)