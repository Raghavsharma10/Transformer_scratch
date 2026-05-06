def add_github_roles(app, repo):
    """Add ``gh`` role to your sphinx documents. It can generate GitHub
    links easily::

        :gh:`issue#57` will generate the issue link
        :gh:`PR#85` will generate the pull request link

    Use this function in ``conf.py`` to enable this feature::

        def setup(app):
            sphinx_typlog_theme.add_github_roles(app, 'lepture/authlib')

    :param app: sphinx app
    :param repo: GitHub repo, e.g. "lepture/authlib"
    """
    from docutils.nodes import reference
    from docutils.parsers.rst.roles import set_classes

    base_url = 'https://github.com/{}'.format(repo)

    def github_role(name, rawtext, text, lineno, inliner,
                    options=None, content=None):
        if '#' in text:
            t, n = text.split('#', 1)
            if t.lower() in ['issue', 'issues']:
                url = base_url + '/issues/{}'.format(n)
            elif t.lower() in ['pr', 'pull', 'pull request']:
                url = base_url + '/pull/{}'.format(n)
            elif t.lower() in ['commit', 'commits']:
                url = base_url + '/commit/{}'.format(n)
        else:
            url = base_url + '/' + text

        options = options or {'classes': ['gh']}
        set_classes(options)
        node = reference(rawtext, text, refuri=url, **options)
        return [node], []

    app.add_role('gh', github_role)