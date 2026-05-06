def pip_command_output(pip_args):
    """
    Get output (as a string) from pip command
    :param pip_args: list o pip switches to pass
    :return: string with results
    """
    import sys
    import pip
    from io import StringIO
    # as pip will write to stdout we use some nasty hacks
    # to substitute system stdout with our own
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    pip.main(pip_args)
    output = mystdout.getvalue()
    mystdout.truncate(0)
    sys.stdout = old_stdout
    return output