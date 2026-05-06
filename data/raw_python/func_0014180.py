def run_command(*args, raise_exception=True, cwd=None):
    '''
    Runs a command, piping all output to the DMP log.
    The args should be separate arguments so paths and subcommands can have spaces in them:

        ret = run_command('ls', '-l', '/Users/me/My Documents')
        print(ret.code)
        print(ret.stdout)
        print(ret.stderr)

    On Windows, the PATH is not followed.  This can be overcome with:

        import shutil
        run_command(shutil.which('program'), '-l', '/Users/me/My Documents')
    '''
    args = [ str(a) for a in args ]
    log.info('running %s', ' '.join(args))
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, cwd=cwd)
    stdout, stderr = p.communicate()
    returninfo = ReturnInfo(p.returncode, stdout.decode('utf8'), stderr.decode('utf8'))
    if stdout:
        log.info('%s', returninfo.stdout)
    if raise_exception and returninfo.code != 0:
        raise CommandError(' '.join(args), returninfo)
    return returninfo