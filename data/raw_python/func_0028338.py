def get_version(root):
    """
    Load and return the contents of version.json.

    :param root: The root path that the ``version.json`` file will be opened
    :type root: str
    :returns: Content of ``version.json`` or None
    :rtype: dict or None
    """
    version_json = os.path.join(root, 'version.json')
    if os.path.exists(version_json):
        with open(version_json, 'r') as version_json_file:
            return json.load(version_json_file)
    return None