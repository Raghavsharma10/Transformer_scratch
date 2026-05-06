def make_semver(version_str):
    """Make a semantic version from Python PEP440 version.

    Semantic versions does not handle post-releases.
    """
    v = parse_version(version_str)

    major = v._version.release[0]
    try:
        minor = v._version.release[1]
    except IndexError:
        minor = 0
    try:
        patch = v._version.release[2]
    except IndexError:
        patch = 0

    prerelease = []
    if v._version.pre:
        prerelease.append(''.join(str(x) for x in v._version.pre))
    if v._version.dev:
        prerelease.append(''.join(str(x) for x in v._version.dev))
    prerelease = '.'.join(prerelease)

    # Create semver
    version = '{0}.{1}.{2}'.format(major, minor, patch)

    if prerelease:
        version += '-{0}'.format(prerelease)
    if v.local:
        version += '+{0}'.format(v.local)

    return version