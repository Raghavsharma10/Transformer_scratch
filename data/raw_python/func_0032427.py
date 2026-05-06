def autoexec(pipe=None, name=None, exit_handler=None):
    """
    create a pipeline with a context that will automatically execute the
    pipeline upon leaving the context if no exception was raised.

    :param pipe:
    :param name:
    :return:
    """
    return pipeline(pipe=pipe, name=name, autoexec=True,
                    exit_handler=exit_handler)