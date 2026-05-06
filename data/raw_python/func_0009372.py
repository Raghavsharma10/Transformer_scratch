def _release_and_issues(changelog, branch, release_type):
    """
    Return most recent branch-appropriate release, if any, and its contents.

    :param dict changelog:
        Changelog contents, as returned by ``releases.util.parse_changelog``.

    :param str branch:
        Branch name.

    :param release_type:
        Member of `Release`, e.g. `Release.FEATURE`.

    :returns:
        Two-tuple of release (``str``) and issues (``list`` of issue numbers.)

        If there is no latest release for the given branch (e.g. if it's a
        feature or master branch), it will be ``None``.
    """
    # Bugfix lines just use the branch to find issues
    bucket = branch
    # Features need a bit more logic
    if release_type is Release.FEATURE:
        bucket = _latest_feature_bucket(changelog)
    # Issues is simply what's in the bucket
    issues = changelog[bucket]
    # Latest release is undefined for feature lines
    release = None
    # And requires scanning changelog, for bugfix lines
    if release_type is Release.BUGFIX:
        versions = [text_type(x) for x in _versions_from_changelog(changelog)]
        release = [x for x in versions if x.startswith(bucket)][-1]
    return release, issues