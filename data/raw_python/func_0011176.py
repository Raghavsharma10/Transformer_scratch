def ls(topics):
    """List contents of current context

    \b
    Usage:
        $ be ls
        - spiderman
        - hulk
        $ be ls spiderman
        - peter
        - mjay
        $ be ls spiderman seq01
        - 1000
        - 2000
        - 2500

    Return codes:
        0 Normal
        2 When insufficient arguments are supplied,
            or a template is unsupported.

    """

    if self.isactive():
        lib.echo("ERROR: Exit current project first")
        sys.exit(lib.USER_ERROR)

    # List projects
    if len(topics) == 0:
        for project in lib.list_projects(root=_extern.cwd()):
            lib.echo("- %s (project)" % project)
        sys.exit(lib.NORMAL)

    # List inventory of project
    elif len(topics) == 1:
        inventory = _extern.load_inventory(topics[0])
        for item, binding in lib.list_inventory(inventory):
            lib.echo("- %s (%s)" % (item, binding))
        sys.exit(lib.NORMAL)

    # List specific portion of template
    else:
        try:
            project = topics[0]
            be = _extern.load_be(project)
            templates = _extern.load_templates(project)
            inventory = _extern.load_inventory(project)
            for item in lib.list_template(root=_extern.cwd(),
                                          topics=topics,
                                          templates=templates,
                                          inventory=inventory,
                                          be=be):
                lib.echo("- %s" % item)
        except IndexError as exc:
            lib.echo(exc)
            sys.exit(lib.USER_ERROR)

    sys.exit(lib.NORMAL)