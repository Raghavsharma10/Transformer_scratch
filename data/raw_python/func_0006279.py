def _extract_version(version_string, pattern):
    """ Extract the version from `version_string` using `pattern`.

    Return the version as a string, with leading/trailing whitespace
    stripped.

    """
    if version_string:
        match = pattern.match(version_string.strip())
        if match:
            return match.group(1)
    return ""