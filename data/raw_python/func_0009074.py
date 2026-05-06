def download_repo(repo_url, destination, commit=None):
    '''download_repo
    :param repo_url: the url of the repo to clone from
    :param destination: the full path to the destination for the repo
    '''
    command = "git clone %s %s" % (repo_url, destination)
    os.system(command)
    return destination