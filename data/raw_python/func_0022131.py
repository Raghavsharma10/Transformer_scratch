def kill_process(procname, scriptname):
    """kill WSGI processes that may be running in development"""

    # from http://stackoverflow.com/a/2940878
    import signal
    import subprocess

    p = subprocess.Popen(['ps', 'aux'], stdout=subprocess.PIPE)
    out, err = p.communicate()

    for line in out.decode().splitlines():
        if procname in line and scriptname in line:
            pid = int(line.split()[1])
            info('Stopping %s %s %d' % (procname, scriptname, pid))
            os.kill(pid, signal.SIGKILL)