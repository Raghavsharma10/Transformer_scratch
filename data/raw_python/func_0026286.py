def _license_from_classifiers(data):
    """try to get a license from the classifiers"""
    classifiers = data.get('classifiers', [])
    found_license = None
    for c in classifiers:
        if c.startswith("License :: OSI Approved :: "):
            found_license = c.replace("License :: OSI Approved :: ", "")
    return found_license