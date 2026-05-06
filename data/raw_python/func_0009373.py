def _get_tags(c):
    """
    Return sorted list of release-style tags as semver objects.
    """
    tags_ = []
    for tagstr in c.run("git tag", hide=True).stdout.strip().split("\n"):
        try:
            tags_.append(Version(tagstr))
        # Ignore anything non-semver; most of the time they'll be non-release
        # tags, and even if they are, we can't reason about anything
        # non-semver anyways.
        # TODO: perhaps log these to DEBUG
        except ValueError:
            pass
    # Version objects sort semantically
    return sorted(tags_)