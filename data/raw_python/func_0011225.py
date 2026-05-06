def echo(text, silent=False, newline=True):
    """Print to the console

    Arguments:
        text (str): Text to print to the console
        silen (bool, optional): Whether or not to produce any output
        newline (bool, optional): Whether or not to append a newline.

    """

    if silent:
        return
    print(text) if newline else sys.stdout.write(text)