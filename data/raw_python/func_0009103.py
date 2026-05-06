def get_image_path(repo_url, trailing_path):
    '''get_image_path will determine an image path based on a repo url, removing
    any token, and taking into account urls that end with .git.
    :param repo_url: the repo url to parse:
    :param trailing_path: the trailing path (commit then hash is common)
    '''
    repo_url = repo_url.split('@')[-1].strip()
    if repo_url.endswith('.git'):
        repo_url =  repo_url[:-4]
    return "%s/%s" %(re.sub('^http.+//www[.]','',repo_url), trailing_path)