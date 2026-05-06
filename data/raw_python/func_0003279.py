def watch_context(keys, result, reqid, container, module = 'objectdb'):
    """
    DEPRECATED - use request_context for most use cases
    """
    try:
        keys = [k for k,r in zip(keys, result) if r is not None]
        yield result
    finally:
        if keys:
            async def clearup():
                try:
                    await send_api(container, module, 'munwatch', {'keys': keys, 'requestid': reqid})
                except QuitException:
                    pass
            container.subroutine(clearup(), False)