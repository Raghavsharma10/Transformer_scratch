async def call_api(container, targetname, name, params = {}, timeout = 120.0):
    """
    Call module API `targetname/name` with parameters.
    
    :param targetname: module targetname. Usually the lower-cased name of the module class, or 'public' for
                       public APIs.
    
    :param name: method name
    
    :param params: module API parameters, should be a dictionary of `{parameter: value}`
    
    :param timeout: raise an exception if the API call is not returned for a long time
    
    :return: API return value
    """
    handle = object()
    apiEvent = ModuleAPICall(handle, targetname, name, params = params)
    await container.wait_for_send(apiEvent)
    replyMatcher = ModuleAPIReply.createMatcher(handle)
    timeout_, ev, m = await container.wait_with_timeout(timeout, replyMatcher)
    if timeout_:
        # Ignore the Event
        apiEvent.canignore = True
        container.scheduler.ignore(ModuleAPICall.createMatcher(handle))
        raise ModuleAPICallTimeoutException('API call timeout')
    else:
        return get_api_result(ev)