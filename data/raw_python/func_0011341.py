def bump_version(version, bump='patch'):
    """patch: patch, minor, major"""
    try:
        parts = map(int, version.split('.'))
    except ValueError:
        fail('Current version is not numeric')

    if bump == 'patch':
        parts[2] += 1
    elif bump == 'minor':
        parts[1] += 1
        parts[2] = 0
    elif bump == 'major':
        parts[0] +=1
        parts[1] = 0
        parts[2] = 0

    return '.'.join(map(str, parts))