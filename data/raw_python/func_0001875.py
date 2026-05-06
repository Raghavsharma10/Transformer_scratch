def contributors(lancet, output):
    """
    List all contributors visible in the git history.
    """
    sorting = pygit2.GIT_SORT_TIME | pygit2.GIT_SORT_REVERSE
    commits = lancet.repo.walk(lancet.repo.head.target, sorting)
    contributors = ((c.author.name, c.author.email) for c in commits)
    contributors = OrderedDict(contributors)

    template_content = content_from_path(
        lancet.config.get('packaging', 'contributors_template'))
    template = Template(template_content)
    output.write(template.render(contributors=contributors).encode('utf-8'))