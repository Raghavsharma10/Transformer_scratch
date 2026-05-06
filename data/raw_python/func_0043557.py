def main():
    """The command line interface of the ``vcs-tool`` program."""
    # Initialize logging to the terminal.
    coloredlogs.install()
    # Command line option defaults.
    repository = None
    revision = None
    actions = []
    # Parse the command line arguments.
    try:
        options, arguments = getopt.gnu_getopt(sys.argv[1:], 'r:dnisume:vqh', [
            'repository=', 'rev=', 'revision=', 'release=', 'find-directory',
            'find-revision-number', 'find-revision-id', 'list-releases',
            'select-release=', 'sum-revisions', 'vcs-control-field', 'update',
            'merge-up', 'export=', 'verbose', 'quiet', 'help',
        ])
        for option, value in options:
            if option in ('-r', '--repository'):
                value = value.strip()
                assert value, "Please specify the name of a repository! (using -r, --repository)"
                repository = coerce_repository(value)
            elif option in ('--rev', '--revision'):
                revision = value.strip()
                assert revision, "Please specify a nonempty revision string!"
            elif option == '--release':
                # TODO Right now --release and --merge-up cannot be combined
                #      because the following statements result in a global
                #      revision id which is immutable. If release objects had
                #      something like an optional `mutable_revision_id' it
                #      should be possible to support the combination of
                #      --release and --merge-up.
                assert repository, "Please specify a repository first!"
                release_id = value.strip()
                assert release_id in repository.releases, "The given release identifier is invalid!"
                revision = repository.releases[release_id].revision.revision_id
            elif option in ('-d', '--find-directory'):
                assert repository, "Please specify a repository first!"
                actions.append(functools.partial(print_directory, repository))
            elif option in ('-n', '--find-revision-number'):
                assert repository, "Please specify a repository first!"
                actions.append(functools.partial(print_revision_number, repository, revision))
            elif option in ('-i', '--find-revision-id'):
                assert repository, "Please specify a repository first!"
                actions.append(functools.partial(print_revision_id, repository, revision))
            elif option == '--list-releases':
                assert repository, "Please specify a repository first!"
                actions.append(functools.partial(print_releases, repository))
            elif option == '--select-release':
                assert repository, "Please specify a repository first!"
                release_id = value.strip()
                assert release_id, "Please specify a nonempty release identifier!"
                actions.append(functools.partial(print_selected_release, repository, release_id))
            elif option in ('-s', '--sum-revisions'):
                assert len(arguments) >= 2, "Please specify one or more repository/revision pairs!"
                actions.append(functools.partial(print_summed_revisions, arguments))
                arguments = []
            elif option == '--vcs-control-field':
                assert repository, "Please specify a repository first!"
                actions.append(functools.partial(print_vcs_control_field, repository, revision))
            elif option in ('-u', '--update'):
                assert repository, "Please specify a repository first!"
                actions.append(functools.partial(repository.update))
            elif option in ('-m', '--merge-up'):
                assert repository, "Please specify a repository first!"
                actions.append(functools.partial(
                    repository.merge_up,
                    target_branch=revision,
                    feature_branch=arguments[0] if arguments else None,
                ))
            elif option in ('-e', '--export'):
                directory = value.strip()
                assert repository, "Please specify a repository first!"
                assert directory, "Please specify the directory where the revision should be exported!"
                actions.append(functools.partial(repository.export, directory, revision))
            elif option in ('-v', '--verbose'):
                coloredlogs.increase_verbosity()
            elif option in ('-q', '--quiet'):
                coloredlogs.decrease_verbosity()
            elif option in ('-h', '--help'):
                usage(__doc__)
                return
        if not actions:
            usage(__doc__)
            return
    except Exception as e:
        warning("Error: %s", e)
        sys.exit(1)
    # Execute the requested action(s).
    try:
        for action in actions:
            action()
    except Exception:
        logger.exception("Failed to execute requested action(s)!")
        sys.exit(1)