def run(cmd, shell=False, debug=False):
    'Run a command and return the output.'
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=shell)
    (out, _) = proc.communicate()  # no need for stderr
    if debug:
        print(cmd)
        print(out)
    return out