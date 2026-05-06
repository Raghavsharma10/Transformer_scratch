def get_provider_manager(osid, runtime=None, proxy=None, local=False):
    """
    Gets the most appropriate provider manager depending on config.

    If local is True, then don't bother with the runtime/config and
    try to get the requested service manager directly from the local
    service implementations known to this mongodb implementation.

    """
    if runtime is not None:
        if local:
            parameter_id = Id('parameter:localImpl@json')
        else:
            parameter_id = Id('parameter:' + osid.lower() + 'ProviderImpl@json')
        try:
            # Try to get the manager from the runtime, if available:
            config = runtime.get_configuration()
            impl_name = config.get_value_by_parameter(parameter_id).get_string_value()
            if proxy is None:
                return runtime.get_manager(osid, impl_name)
            else:
                return runtime.get_proxy_manager(osid, impl_name)
        except (AttributeError, KeyError, NotFound):
            pass
    # Try to return a Manager directly from this implementation, or raise OperationFailed:
    try:
        if proxy is None:
            mgr_str = 'Manager'
        else:
            mgr_str = 'ProxyManager'
        module = import_module(
            'dlkit.json_.' + fix_reserved_word(osid.lower(), is_module=True) + '.managers')
        manager_name = ''.join((osid.title()).split('_')) + mgr_str
        manager = getattr(module, manager_name)()
    except (ImportError, AttributeError):
        raise OperationFailed()
    if runtime is not None:
        manager.initialize(runtime)
    return manager