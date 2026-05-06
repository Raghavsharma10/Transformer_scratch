async def send_api(container, targetname, name, params = {}):
    """
    Send API and discard the result
    """
    handle = object()
    apiEvent = ModuleAPICall(handle, targetname, name, params = params)
    await container.wait_for_send(apiEvent)