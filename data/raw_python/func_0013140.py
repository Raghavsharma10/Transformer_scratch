def source_foreign(args, stdin=None):
    """Sources a file written in a foreign shell language."""
    parser = _ensure_source_foreign_parser()
    ns = parser.parse_args(args)
    if ns.prevcmd is not None:
        pass  # don't change prevcmd if given explicitly
    elif os.path.isfile(ns.files_or_code[0]):
        # we have filename to source
        ns.prevcmd = '{} "{}"'.format(ns.sourcer, '" "'.join(ns.files_or_code))
    elif ns.prevcmd is None:
        ns.prevcmd = ' '.join(ns.files_or_code)  # code to run, no files
    fsenv = foreign_shell_data(shell=ns.shell, login=ns.login,
                                          interactive=ns.interactive,
                                          envcmd=ns.envcmd,
                                          aliascmd=ns.aliascmd,
                                          extra_args=ns.extra_args,
                                          safe=ns.safe, prevcmd=ns.prevcmd,
                                          postcmd=ns.postcmd,
                                          funcscmd=ns.funcscmd,
                                          sourcer=ns.sourcer,
                                          use_tmpfile=ns.use_tmpfile,
                                          seterrprevcmd=ns.seterrprevcmd,
                                          seterrpostcmd=ns.seterrpostcmd)
    if fsenv is None:
        raise RuntimeError("Source failed: {}\n".format(ns.prevcmd), 1)
    # apply results
    env = os.environ.copy()
    for k, v in fsenv.items():
        if k in env and v == env[k]:
            continue  # no change from original
        env[k] = v
    # Remove any env-vars that were unset by the script.
    for k in os.environ: # use os.environ again to prevent errors about changed size
        if k not in fsenv:
            env.pop(k, None)
    return env