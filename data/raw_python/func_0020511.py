def make_name_from_git(repo, branch, limit=53, separator='-', hash_size=5):
    """
    return name string representing the given git repo and branch
    to be used as a build name.

    NOTE: Build name will be used to generate pods which have a
    limit of 64 characters and is composed as:

        <buildname>-<buildnumber>-<podsuffix>
        rhel7-1-build

    Assuming '-XXXX' (5 chars) and '-build' (6 chars) as default
    suffixes, name should be limited to 53 chars (64 - 11).

    OpenShift is very peculiar in which BuildConfig names it
    allows. For this reason, only certain characters are allowed.
    Any disallowed characters will be removed from repo and
    branch names.

    :param repo: str, the git repository to be used
    :param branch: str, the git branch to be used
    :param limit: int, max name length
    :param separator: str, used to separate the repo and branch in name

    :return: str, name representing git repo and branch.
    """

    branch = branch or 'unknown'
    full = urlparse(repo).path.lstrip('/') + branch
    repo = git_repo_humanish_part_from_uri(repo)
    shaval = sha256(full.encode('utf-8')).hexdigest()
    hash_str = shaval[:hash_size]
    limit = limit - len(hash_str) - 1

    sanitized = sanitize_strings_for_openshift(repo, branch, limit, separator, False)
    return separator.join(filter(None, (sanitized, hash_str)))