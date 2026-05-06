def bash_echo_warmer(n):
    """A very basic example of how to create n additional tasks.
    This is a warmer function with randomly delayed effects on the
    bash_echo_metric and random task lengths to make the metric less
    predictable
    """
    import subprocess
    import random
    cmd = (
        'set -o pipefail '
        " ; sleep %s "
        " ; sh -c 'echo from bash: started relay launcher task && sleep %s'"
    )
    for i in range(n):
        subprocess.Popen(
            cmd % ((1 + random.random()) * 1, (1 + random.random()) * 4),
            shell=True, stdout=subprocess.PIPE, executable='bash')