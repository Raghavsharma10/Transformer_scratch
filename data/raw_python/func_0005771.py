def _match_tags(repex_tags, path_tags):
    """Check for matching tags between what the user provided
    and the tags set in the config.

    If `any` is chosen, match.
    If no tags are chosen and none are configured, match.
    If the user provided tags match any of the configured tags, match.
    """
    if 'any' in repex_tags or (not repex_tags and not path_tags):
        return True
    elif set(repex_tags) & set(path_tags):
        return True
    return False