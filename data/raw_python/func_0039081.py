def compiler_preprocessor_verbose(compiler, extraflags):
    """Capture the compiler preprocessor stage in verbose mode
    """
    lines = []
    with open(os.devnull, 'r') as devnull:
        cmd = [compiler, '-E'] 
        cmd += extraflags
        cmd += ['-', '-v']
        p = Popen(cmd, stdin=devnull, stdout=PIPE, stderr=PIPE)
        p.wait()
        p.stdout.close()
        lines = p.stderr.read()
        lines = lines.decode('utf-8')
        lines = lines.splitlines()
    return lines