def clone_git_repo(repo_url):
    """
    input: repo_url
    output: path of the cloned repository
    steps:
        1. clone the repo
        2. parse 'site' into for templating

    assumptions:
        repo_url = "git@github.com:littleq0903/django-deployer-template-openshift-experiment.git"
        repo_local_location = "/tmp/djangodeployer-cache-xxxx" # xxxx here will be some short uuid for identify different downloads
    """
    REPO_PREFIX = "djangodeployer-cache-"
    REPO_POSTFIX_UUID = str(uuid.uuid4()).split('-')[-1]
    REPO_CACHE_NAME = REPO_PREFIX + REPO_POSTFIX_UUID
    REPO_CACHE_LOCATION = '/tmp/%s' % REPO_CACHE_NAME

    repo = git.Repo.clone_from(repo_url, REPO_CACHE_LOCATION)
    return REPO_CACHE_LOCATION