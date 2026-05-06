def runcmd(command, command_input=None, cwd=None):
    """Run a command, potentially sending stdin, and capturing stdout/err."""
    proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            cwd=cwd)
    (stdout, stderr) = proc.communicate(command_input)
    if proc.returncode != 0:
        sys.stderr.write('ABORTING: command "%s" failed w/ code %s:\n'
                         '%s\n%s' % (command, proc.returncode,
                                     stdout, stderr))
        sys.exit(proc.returncode)
    return proc.returncode, stdout, stderr