def monkey_patch_issue_25593():
    """ Workaround for http://bugs.python.org/issue25593 """
    save = asyncio.selector_events.BaseSelectorEventLoop._sock_connect_cb

    @functools.wraps(save)
    def patched(instance, fut, sock, address):
        if not fut.done():
            save(instance, fut, sock, address)
    asyncio.selector_events.BaseSelectorEventLoop._sock_connect_cb = patched