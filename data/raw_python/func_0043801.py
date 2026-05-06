def find_site(site, path=None):
    " Return inited site by name (project.bracnch) or path "

    try:
        return Site(site)

    except AssertionError:
        path = path or settings.MAKESITE_HOME
        if op.sep in site:
            raise

        site = site if '.' in site else "%s.master" % site
        project, branch = site.split('.', 1)
        return Site(op.join(path, project, branch))