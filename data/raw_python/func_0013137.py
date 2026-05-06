def source_cmd(args, stdin=None):
    """Simple cmd.exe-specific wrapper around source-foreign.

    returns a dict to be used as a new environment
    """
    args = list(args)
    fpath = locate_binary(args[0])
    args[0] = fpath if fpath else args[0]
    if not os.path.isfile(args[0]):
        raise RuntimeError("Command not found: %s" % args[0])
    prevcmd = 'call '
    prevcmd += ' '.join([argvquote(arg, force=True) for arg in args])
    prevcmd = escape_windows_cmd_string(prevcmd)
    args.append('--prevcmd={}'.format(prevcmd))
    args.insert(0, 'cmd')
    args.append('--interactive=0')
    args.append('--sourcer=call')
    args.append('--envcmd=set')
    args.append('--seterrpostcmd=if errorlevel 1 exit 1')
    args.append('--use-tmpfile=1')
    return source_foreign(args, stdin=stdin)