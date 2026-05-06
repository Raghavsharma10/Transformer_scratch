def start_daemon_in_subprocess(options, outpath=os.devnull):
    """
    Run `rash daemon --no-error` in background.

    :type options: list of str
    :arg  options: options for "rash daemon" command
    :type outpath: str
    :arg  outpath: path to redirect daemon output

    """
    import subprocess
    import sys
    from .utils.py3compat import nested
    from .utils.pathutils import mkdirp
    if outpath != os.devnull:
        mkdirp(os.path.dirname(outpath))
    with nested(open(os.devnull),
                open(outpath, 'w')) as (stdin, stdout):
        subprocess.Popen(
            [os.path.abspath(sys.executable), '-m', 'rash.cli',
             'daemon', '--no-error'] + options,
            preexec_fn=os.setsid,
            stdin=stdin, stdout=stdout, stderr=subprocess.STDOUT)