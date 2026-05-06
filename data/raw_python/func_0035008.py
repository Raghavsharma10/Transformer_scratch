def print_pretty_command(env, command):
    """This is a hack for prettier printing.

    Rather than "{envpython} foo.py" we print "python foo.py".

    """
    cmd = abbr_cmd = command[0]
    if cmd.startswith(env.envbindir):
        abbr_cmd = os.path.relpath(cmd, env.envbindir)
        if abbr_cmd == ".":
            # TODO are there more edge cases?
            abbr_cmd = cmd
    command[0] = abbr_cmd
    print('(%s)$ %s' % (env.name, ' '.join(['"%s"' % c if " " in c
                                            else c
                                            for c in command])))
    command[0] = cmd
    return abbr_cmd, cmd, command