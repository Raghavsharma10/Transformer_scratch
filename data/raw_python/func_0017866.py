def main():
    """
    The main entry point for datacats cli tool

    (as defined in setup.py's entry_points)
    It parses the cli arguments for corresponding options
    and runs the corresponding command
    """
    # pylint: disable=bare-except
    try:
        command_fn, opts = _parse_arguments(sys.argv[1:])
        # purge handles loading differently
        # 1 - Bail and just call the command if it doesn't have ENVIRONMENT.
        if command_fn == purge.purge or 'ENVIRONMENT' not in opts:
            return command_fn(opts)

        environment = Environment.load(
            opts['ENVIRONMENT'] or '.',
            opts['--site'] if '--site' in opts else 'primary')

        if command_fn not in COMMANDS_THAT_USE_SSH:
            return command_fn(environment, opts)

        # for commands that communicate with a remote server
        # we load UserProfile and test our communication
        user_profile = UserProfile()
        user_profile.test_ssh_key(environment)

        return command_fn(environment, opts, user_profile)

    except DatacatsError as e:
        _error_exit(e)
    except SystemExit:
        raise
    except:
        exc_info = "\n".join([line.rstrip()
            for line in traceback.format_exception(*sys.exc_info())])
        user_message = ("Something that should not"
            " have happened happened when attempting"
            " to run this command:\n"
            "     datacats {args}\n\n"
            "It is seems to be a bug.\n"
            "Please report this issue to us by"
            " creating an issue ticket at\n\n"
            "    https://github.com/datacats/datacats/issues\n\n"
            "so that we would be able to look into that "
            "and fix the issue."
            ).format(args=" ".join(sys.argv[1:]))

        _error_exit(DatacatsError(user_message,
            parent_exception=UndocumentedError(exc_info)))