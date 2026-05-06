def compare_version(version1, version2):
    """
    Compares two versions.
    """
    def normalize(v):
        return [int(x) for x in re.sub(r'(\.0+)*$','', v).split(".")]
    return (normalize(version1) > normalize(version2))-(normalize(version1) < normalize(version2))