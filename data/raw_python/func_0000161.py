def get_related_targetApplication(vR, app_id, app_ver):
    """Return the first matching target application in this version range.
    Returns None if there are no target applications or no matching ones."""
    targetApplication = vR.get('targetApplication')
    if not targetApplication:
        return None

    for tA in targetApplication:
        guid = tA.get('guid')
        if not guid or guid == app_id:
            if not app_ver:
                return tA
            # We purposefully use maxVersion only, so that the blocklist contains items
            # whose minimum version is ahead of the version we get passed. This means
            # the blocklist we serve is "future-proof" for app upgrades.
            if between(version_int(app_ver), '0', tA.get('maxVersion', '*')):
                return tA

    return None