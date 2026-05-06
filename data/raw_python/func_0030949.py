def get_url_file_name(url):
    """Get the file name from an url
    
    Parameters
    ----------
    url : str

    Returns
    -------
    str
        The file name 
    """

    assert isinstance(url, (str, _oldstr))
    return urlparse.urlparse(url).path.split('/')[-1]