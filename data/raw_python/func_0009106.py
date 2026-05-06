def get_build_metadata(key):
    '''get_build_metadata will return metadata about an instance from within it.
    :param key: the key to look up
    '''
    headers = {"Metadata-Flavor":"Google"}
    url = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/%s" %(key)        
    response = requests.get(url=url,headers=headers)
    if response.status_code == 200:
        return response.text
    return None