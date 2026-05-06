def call_pre_hook(awsclient, cloudformation):
    """Invoke the pre_hook BEFORE the config is read.

    :param awsclient:
    :param cloudformation:
    """
    # TODO: this is deprecated!! move this to glomex_config_reader
    # no config available
    if not hasattr(cloudformation, 'pre_hook'):
        # hook is not present
        return
    hook_func = getattr(cloudformation, 'pre_hook')
    if not hook_func.func_code.co_argcount:
        hook_func()  # for compatibility with existing templates
    else:
        log.error('pre_hock can not have any arguments. The pre_hook it is ' +
                  'executed BEFORE config is read')