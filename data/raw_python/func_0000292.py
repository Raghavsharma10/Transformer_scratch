def semantic_version(tag):
    """Get a valid semantic version for tag
    """
    try:
        version = list(map(int, tag.split('.')))
        assert len(version) == 3
        return tuple(version)
    except Exception as exc:
        raise CommandError(
            'Could not parse "%s", please use '
            'MAJOR.MINOR.PATCH' % tag
        ) from exc