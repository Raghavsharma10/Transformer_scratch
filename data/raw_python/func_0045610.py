def run(command, timeout=None, cwd=None, env=None, debug=None):
    """
    Runs a given command on the system within a set time period, providing an easy way to access
    command output as it happens without waiting for the command to finish running.

    :type list
    :param command: Should be a list that contains the command that should be ran on the given
                    system. The only whitespaces that can occur is for paths that use a backslash
                    to escape it appropriately

    :type int
    :param timeout: Specificed in seconds. If a command outruns the timeout then the command and
                    its child processes will be terminated. The default is to run

    :type string
    :param cwd: If cwd is set then the current directory will be changed to cwd before it is executed.
                Note that this directory is not considered when searching the executable, so you
                can’t specify the program’s path relative to cwd.

    :type dict
    :param env: A dict of any ENV variables that should be combined into the OS ENV that will help
                the command to run successfully. Note that more often than not the command run
                does not have the same ENV variables available as your shell by default and as such
                require some assistance.

    :type function
    :param debug: A function (also a class function) can be passed in here and all output, line by line,
                  from the command being run will be passed to it as it gets outputted to stdout.
                  This allows for things such as logging (using the built in python logging lib)
                  what is happening on long running commands or redirect output of a tail -f call
                  as lines get outputted without having to wait till the command finishes.

    :return returns :class:`Command.Response` that contains the exit code and the output from the command
    """
    return Command.run(command, timeout=timeout, cwd=cwd, env=env, debug=debug)