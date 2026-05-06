def get_dependency_metadata():
    '''Returns list of strings with dependency metadata from Dapi'''
    link = os.path.join(_api_url(), 'meta.txt')
    return _process_req_txt(requests.get(link)).split('\n')