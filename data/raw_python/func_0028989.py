def _load_hooks(path):
    """Load hook module and register signals.

    :param path: Absolute or relative path to module.
    :return: module
    """
    module = imp.load_source(os.path.splitext(os.path.basename(path))[0], path)
    if not check_hook_mechanism_is_intact(module):
        # no hooks - do nothing
        log.debug('No valid hook configuration: \'%s\'. Not using hooks!', path)
    else:
        if check_register_present(module):
            # register the template hooks so they listen to gcdt_signals
            module.register()
    return module