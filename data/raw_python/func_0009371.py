def _versions_from_changelog(changelog):
    """
    Return all released versions from given ``changelog``, sorted.

    :param dict changelog:
        A changelog dict as returned by ``releases.util.parse_changelog``.

    :returns: A sorted list of `semantic_version.Version` objects.
    """
    versions = [Version(x) for x in changelog if BUGFIX_RELEASE_RE.match(x)]
    return sorted(versions)