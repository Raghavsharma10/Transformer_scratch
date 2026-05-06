def get_version(version):
    """
    Returns a PEP 440-compliant version number from VERSION.

    Created by modifying django.utils.version.get_version
    """

    # Now build the two parts of the version number:
    # major = X.Y[.Z]
    # sub = .devN - for development releases
    #     | {a|b|rc}N - for alpha, beta and rc releases
    #     | .postN - for post-release releases

    assert len(version) == 5

    version_parts = version[:2] if version[2] == 0 else version[:3]

    # Build the first part of the version
    major = '.'.join(str(x) for x in version_parts)

    # Just return it if this is a final release version
    if version[3] == 'final':
        return major

    # Add the rest
    sub = ''.join(str(x) for x in version[3:5])

    if version[3] == 'dev':
        # Override the sub part.  Add in a timestamp
        timestamp = get_git_changeset()
        sub = 'dev%s' % (timestamp if timestamp else version[4])
        return '%s.%s' % (major, sub)
    if version[3] == 'post':
        # We need a dot for post
        return '%s.%s' % (major, sub)
    elif version[3] in ('a', 'b', 'rc'):
        # No dot for these
        return '%s%s' % (major, sub)
    else:
        raise ValueError('Invalid version: %s' % str(version))