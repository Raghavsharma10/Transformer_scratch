def git_day():
    """Constructs a version string of the form:

           day[.<commit-number-in-day>][+<branch-name-if-not-master>]

       Master is understood to be always buildable and thus untagged
       versions are treated as patch levels. Branches not master are treated
       as PEP-440 "local version identifiers".
    """
    vec = ['env', 'TZ=UTC', 'git', 'log', '--date=iso-local', '--pretty=%ad']
    day = cmd(*(vec + ['-n', '1'])).split()[0]
    commits = cmd(*(vec + ['--since', day + 'T00:00Z'])).strip()
    n = len(commits.split('\n'))
    day = day.replace('-', '')
    if n > 1:
        day += '.%s' % n
    # Branches that are not master are treated as local:
    #   https://www.python.org/dev/peps/pep-0440/#local-version-identifiers
    branch = get_git_branch()
    if branch != 'master':
        day += '+' + s(branch)
    return day