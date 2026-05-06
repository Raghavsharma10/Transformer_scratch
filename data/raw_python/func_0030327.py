def bash_echo_metric():
    """A very basic example that monitors
    a number of currently running processes"""
    import subprocess
    # import random

    # more predictable version of the metric
    cmd = (
        'set -o pipefail '
        ' ; pgrep -f "^bash.*sleep .*from bash: started relay launcher"'
        ' | wc -l '
    )

    # less predictable version of the metric
    # cmd = 'ps aux|wc -l'

    while True:
        yield (
            int(subprocess.check_output(cmd, shell=True, executable='bash'))
            # + random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        )