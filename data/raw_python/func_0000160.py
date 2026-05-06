def is_related_to(item, app_id, app_ver=None):
    """Return True if the item relates to the given app_id (and app_ver, if passed)."""
    versionRange = item.get('versionRange')
    if not versionRange:
        return True

    for vR in versionRange:
        if not vR.get('targetApplication'):
            return True
        if get_related_targetApplication(vR, app_id, app_ver) is not None:
            return True
    return False