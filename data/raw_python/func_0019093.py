def run_subprocess(command: str, verbose: bool = True, blocking: bool = True) \
        -> Optional[subprocess.Popen]:
    """Execute the given command in a new process.

    Only when both `verbose` and `blocking` are |True|, |run_subprocess|
    prints all responses to the current value of |sys.stdout|:

    >>> from hydpy import run_subprocess
    >>> import platform
    >>> esc = '' if 'windows' in platform.platform().lower() else '\\\\'
    >>> run_subprocess(f'python -c print{esc}(1+1{esc})')
    2

    With verbose being |False|, |run_subprocess| does never print out
    anything:

    >>> run_subprocess(f'python -c print{esc}(1+1{esc})', verbose=False)

    >>> process = run_subprocess('python', blocking=False, verbose=False)
    >>> process.kill()
    >>> _ = process.communicate()

    When `verbose` is |True| and `blocking` is |False|, |run_subprocess|
    prints all responses to the console ("invisible" for doctests):

    >>> process = run_subprocess('python', blocking=False)
    >>> process.kill()
    >>> _ = process.communicate()
    """
    if blocking:
        result1 = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            shell=True)
        if verbose:    # due to doctest replacing sys.stdout
            for output in (result1.stdout, result1.stderr):
                output = output.strip()
                if output:
                    print(output)
        return None
    stdouterr = None if verbose else subprocess.DEVNULL
    result2 = subprocess.Popen(
        command,
        stdout=stdouterr,
        stderr=stdouterr,
        encoding='utf-8',
        shell=True)
    return result2