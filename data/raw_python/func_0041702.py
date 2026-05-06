def popen_wrapper(args):
    """
    Friendly wrapper around Popen.

    Returns stdout output, stderr output and OS status code.
    """
    try:
        p = Popen(args,
                  shell=False,
                  stdout=PIPE,
                  stderr=PIPE,
                  close_fds=os.name != 'nt',
                  universal_newlines=True)
    except OSError as e:
        raise OSError(
            "Error executing '{:}': '{:}'".format(args[0], e.strerror))
    output, errors = p.communicate()
    return (
        output,
        text_type(errors),
        p.returncode
    )