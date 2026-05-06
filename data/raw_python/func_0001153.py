def get_arguments(argv=None, environ=None):
    """Get command line arguments or values from environment variables.

    :param list argv: Command line argument list to process. For testing.
    :param dict environ: Environment variables. For testing.

    :return: Parsed options.
    :rtype: dict
    """
    name = 'appveyor-artifacts'
    environ = environ or os.environ
    require = getattr(pkg_resources, 'require')  # Stupid linting error.
    commit, owner, pull_request, repo, tag = '', '', '', '', ''

    # Run docopt.
    project = [p for p in require(name) if p.project_name == name][0]
    version = project.version
    args = docopt(__doc__, argv=argv or sys.argv[1:], version=version)

    # Handle Travis environment variables.
    if environ.get('TRAVIS') == 'true':
        commit = environ.get('TRAVIS_COMMIT', '')
        owner = environ.get('TRAVIS_REPO_SLUG', '/').split('/')[0]
        pull_request = environ.get('TRAVIS_PULL_REQUEST', '')
        if pull_request == 'false':
            pull_request = ''
        repo = environ.get('TRAVIS_REPO_SLUG', '/').split('/')[1].replace('_', '-')
        tag = environ.get('TRAVIS_TAG', '')

    # Command line arguments override.
    commit = args['--commit'] or commit
    owner = args['--owner-name'] or owner
    pull_request = args['--pull-request'] or pull_request
    repo = args['--repo-name'] or repo
    tag = args['--tag-name'] or tag

    # Merge env variables and have command line args override.
    config = {
        'always_job_dirs': args['--always-job-dirs'],
        'commit': commit,
        'dir': args['--dir'] or '',
        'ignore_errors': args['--ignore-errors'],
        'job_name': args['--job-name'] or '',
        'mangle_coverage': args['--mangle-coverage'],
        'no_job_dirs': args['--no-job-dirs'] or '',
        'owner': owner,
        'pull_request': pull_request,
        'raise': args['--raise'],
        'repo': repo,
        'tag': tag,
        'verbose': args['--verbose'],
    }

    return config