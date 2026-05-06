def git_version():
    """Constructs a version string of the form:

           <tag>[.<distance-from-tag>[+<branch-name-if-not-master>]]

       Master is understood to be always buildable and thus untagged
       versions are treated as patch levels. Branches not master are treated
       as PEP-440 "local version identifiers".
    """
    tag = cmd('git', 'describe').strip()
    pieces = s(tag).split('-')
    dotted = pieces[0]
    if len(pieces) < 2:
        distance = None
    else:
        # Distance from the latest tag is treated as a patch level.
        distance = pieces[1]
        dotted += '.' + s(distance)
    # Branches that are not master are treated as local:
    #   https://www.python.org/dev/peps/pep-0440/#local-version-identifiers
    if distance is not None:
        branch = get_git_branch()
        if branch != 'master':
            dotted += '+' + s(branch)
    return dotted