def git_tag_to_semver(git_tag: str) -> SemVer:
    """
    :git_tag: A string representation of a Git tag.

    Searches a Git tag's string representation for a SemVer, and returns that
    as a SemVer object.
    """
    pattern = re.compile(r'[0-9]+\.[0-9]+\.[0-9]+$')
    match = pattern.search(git_tag)
    if match:
        version = match.group(0)
    else:
        raise InvalidTagFormatException('Tag passed contains no SemVer.')

    return SemVer.from_str(version)