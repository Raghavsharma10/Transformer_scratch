def _strip_version_from_dependency(dep):
    '''For given dependency string, return only the package name'''
    usedmark = ''
    for mark in '< > ='.split():
        split = dep.split(mark)
        if len(split) > 1:
            usedmark = mark
            break
    if usedmark:
        return split[0].strip()
    else:
        return dep.strip()