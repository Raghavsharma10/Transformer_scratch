def get_context(awsclient, env, tool, command, arguments=None):
    """This assembles the tool context. Private members are preceded by a '_'.

    :param tool:
    :param command:
    :return: dictionary containing the gcdt tool context
    """
    # TODO: elapsed, artifact(stack, depl-grp, lambda, api)
    if arguments is None:
        arguments = {}
    context = {
        '_awsclient': awsclient,
        'env': env,
        'tool': tool,
        'command': command,
        '_arguments': arguments,  # TODO clean up arguments -> args
        'version': __version__,
        'user': _get_user(),
        'plugins': get_plugin_versions().keys()
    }

    return context