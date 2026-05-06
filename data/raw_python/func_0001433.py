def last_git_release_tag(git_tags: str) -> str:
    """
    :git_tags: chronos.helpers.git_tags() function output.

    Returns the latest Git tag ending with a SemVer as a string.
    """
    semver_re = re.compile(r'[0-9]+\.[0-9]+\.[0-9]+$')
    str_ver = []
    for i in git_tags.split():
        if semver_re.search(i):
            str_ver.append(i)

    try:
        return str_ver[0]
    except IndexError:
        raise NoGitTagsException