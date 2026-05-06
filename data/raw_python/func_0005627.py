def changelog():
    # type: () -> str
    """ Print change log since last release. """
    # Skip 'v' prefix
    versions = [x for x in git.tags() if versioning.is_valid(x[1:])]

    cmd = 'git log --format=%H'
    if versions:
        cmd += ' {}..HEAD'.format(versions[-1])

    hashes = shell.run(cmd, capture=True).stdout.strip().splitlines()
    commits = [git.CommitDetails.get(h) for h in hashes]

    tags = conf.get('changelog.tags', [
        {'header': 'Features', 'tag': 'feature'},
        {'header': 'Changes', 'tag': 'change'},
        {'header': 'Fixes', 'tag': 'fix'},
    ])

    results = OrderedDict((
        (x['header'], []) for x in tags
    ))

    for commit in commits:
        commit_items = extract_changelog_items(commit.desc, tags)
        for header, items in commit_items.items():
            results[header] += items

    lines = [
        '<35>v{}<0>'.format(versioning.current()),
        '',
    ]
    for header, items in results.items():
        if items:
            lines += [
                '',
                '<32>{}<0>'.format(header),
                '<32>{}<0>'.format('-' * len(header)),
                '',
            ]
            for item_text in items:
                item_lines = textwrap.wrap(item_text, 77)
                lines += ['- {}'.format('\n  '.join(item_lines))]

            lines += ['']

    return '\n'.join(lines)