def bash_echo_cooler(n):
    """A very basic example of how to destroy n running tasks
    This is a cooler function
    """
    import subprocess
    cmd = (
        'set -o pipefile '
        ' ; kill `pgrep -f "from bash: started relay launcher task"'
        ' | tail -n %s` 2>/dev/null' % n)
    subprocess.Popen(cmd, shell=True, executable='bash').wait()