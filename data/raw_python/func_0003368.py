async def batch_call_api(container, apis, timeout = 120.0):
    """
    DEPRECATED - use execute_all instead
    """
    apiHandles = [(object(), api) for api in apis]
    apiEvents = [ModuleAPICall(handle, targetname, name, params = params)
                 for handle, (targetname, name, params) in apiHandles]
    apiMatchers = tuple(ModuleAPIReply.createMatcher(handle) for handle, _ in apiHandles)
    async def process():
        for e in apiEvents:
            await container.wait_for_send(e)
    container.subroutine(process(), False)
    eventdict = {}
    async def process2():
        ms = len(apiMatchers)
        matchers = Diff_(apiMatchers)
        while ms:
            ev, m = await matchers
            matchers = Diff_(matchers, remove=(m,))
            eventdict[ev.handle] = ev
    await container.execute_with_timeout(timeout, process2())
    for e in apiEvents:
        if e.handle not in eventdict:
            e.canignore = True
            container.scheduler.ignore(ModuleAPICall.createMatcher(e.handle))
    return [eventdict.get(handle, None) for handle, _ in apiHandles]