def run_process(cwd, args):
    """Executes an external process via subprocess.Popen"""
    try:
        process = check_output(args, cwd=cwd, stderr=STDOUT)

        return process
    except CalledProcessError as e:
        log('Uh oh, the teapot broke again! Error:', e, type(e), lvl=verbose, pretty=True)
        log(e.cmd, e.returncode, e.output, lvl=verbose)
        return e.output