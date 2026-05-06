def ctox(arguments, toxinidir):
    """Sets up conda environments, and sets up and runs each environment based
    on the project's tox.ini configuration file.

    Returns 1 if either the build or running the commands failed or 0 if
    all commmands ran successfully.

    """
    if arguments is None:
        arguments = []
    if toxinidir is None:
        toxinidir = os.getcwd()

    args, options = parse_args(arguments)

    if args.version:
        print(version)
        return 0

    # if no conda trigger OSError
    try:
        with open(os.devnull, "w") as fnull:
            check_output(['conda', '--version'], stderr=fnull)
    except OSError:
        cprint("conda not found, you need to install it to use ctox.\n"
               "The recommended way is to download miniconda,\n"
               "Do not install conda via pip.", 'err')
        return 1

    toxinifile = os.path.join(toxinidir, "tox.ini")

    from ctox.config import read_config, get_envlist
    config = read_config(toxinifile)
    if args.e == 'ALL':
        envlist = get_envlist(config)
    else:
        envlist = args.e.split(',')

    # TODO configure with option
    toxdir = os.path.join(toxinidir, ".tox")

    # create a zip file for the project
    from ctox.pkg import make_dist, package_name
    cprint("GLOB sdist-make: %s" % os.path.join(toxinidir, "setup.py"))
    package = package_name(toxinidir)
    if not make_dist(toxinidir, toxdir, package):
        cprint("    setup.py sdist failed", 'err')
        return 1

    # setup each environment and run ctox
    failing = {}
    for env_name in envlist:
        env = Env(name=env_name, config=config, options=options,
                  toxdir=toxdir, toxinidir=toxinidir, package=package)
        failing[env_name] = env.ctox()

    # print summary of the outcomes of ctox for each environment
    cprint('Summary')
    print("-" * 23)
    for env_name in envlist:
        n = failing[env_name]
        outcome = ('succeeded', 'failed', 'skipped')[n]
        status = ('ok', 'err', 'warn')[n]
        cprint("%s commands %s" % (env_name, outcome), status)

    return any(1 == v for v in failing.values())