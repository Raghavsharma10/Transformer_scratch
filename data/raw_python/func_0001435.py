def parse_commit_log(commit_log: dict) -> str:
    """
    :commit_log: chronos.helpers.git_commits_since_last_tag() output.

    Parse Git log and return either 'maj', 'min', or 'pat'.
    """
    rv = 'pat'

    cc_patterns = patterns()

    for value in commit_log.values():
        if re.search(cc_patterns['feat'], value):
            rv = 'min'
        if re.search(cc_patterns['BREAKING CHANGE'], value):
            rv = 'maj'

    return rv