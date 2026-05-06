def _main(argv):
    """ Function that acts just like main() except
    doesn't catch exceptions. """
    repo_input_argv = len(argv) == 2 and argv[0] in ['--repo', '-r', '-R']

    # We only support a single argv parameter.
    if len(argv) > 1 and not repo_input_argv:
        _main(['--help'])

    # Parse the command and do the right thing.
    if len(argv) == 1 or repo_input_argv:
        arg = argv[0]

        # Help/usage
        if arg in ['-h', '--help', '-H']:
            print(_USAGE)

        # Version
        elif arg in ['-v', '--version', '-V']:
            print(_version_string())

        # Token
        elif arg in ['-r', '--repo', '-R']:
            if len(argv) == 2:
                url = argv[1]
            else:
                url = None
            _input_github_repo(url)

        # No wait
        elif arg in ['--no-wait', '-nw']:
            url = _load_github_repo()
            commit, committed = _submit_changes_to_github_repo(os.getcwd(),
                                                               url)
            build_id = _wait_for_travis_build(url, commit, committed)

        # Help string
        else:
            _main(['--help'])

    # No arguments means we're trying to submit to Travis.
    elif len(argv) == 0:
        url = _load_github_repo()
        commit, committed = _submit_changes_to_github_repo(os.getcwd(), url)
        build_id = _wait_for_travis_build(url, commit, committed)
        _watch_travis_build(build_id)