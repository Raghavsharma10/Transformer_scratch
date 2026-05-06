def docs():
    """ Create documentation.
    """
    from epydoc import cli

    path('build').exists() or path('build').makedirs()

    # get storage path
    docs_dir = options.docs.get('docs_dir', 'docs/apidocs')

    # clean up previous docs
    (path(docs_dir) / "epydoc.css").exists() and path(docs_dir).rmtree()

    # set up excludes
    try:
        exclude_names = options.docs.excludes
    except AttributeError:
        exclude_names = []
    else:
        exclude_names = exclude_names.replace(',', ' ').split()

    excludes = []
    for pkg in exclude_names:
        excludes.append("--exclude")
        excludes.append('^' + re.escape(pkg))

    # call epydoc in-process
    sys_argv = sys.argv
    try:
        sys.argv = [
            sys.argv[0] + "::epydoc",
            "-v",
            "--inheritance", "listed",
            "--output", docs_dir,
            "--name", "%s %s" % (options.setup.name, options.setup.version),
            "--url", options.setup.url,
            "--graph", "umlclasstree",
        ] + excludes + toplevel_packages()
        sys.stderr.write("Running '%s'\n" % ("' '".join(sys.argv)))
        cli.cli()
    finally:
        sys.argv = sys_argv