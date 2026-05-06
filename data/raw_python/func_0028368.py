def asyncio_run_forever(setup_coro, shutdown_coro, *,
                        stop_signals={signal.SIGINT}, debug=False):
    '''
    A proposed-but-not-implemented asyncio.run_forever() API based on
    @vxgmichel's idea.
    See discussions on https://github.com/python/asyncio/pull/465
    '''
    async def wait_for_stop():
        loop = current_loop()
        future = loop.create_future()
        for stop_sig in stop_signals:
            loop.add_signal_handler(stop_sig, future.set_result, stop_sig)
        try:
            recv_sig = await future
        finally:
            loop.remove_signal_handler(recv_sig)

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.set_debug(debug)
        loop.run_until_complete(setup_coro)
        loop.run_until_complete(wait_for_stop())
    finally:
        try:
            loop.run_until_complete(shutdown_coro)
            _cancel_all_tasks(loop)
            if hasattr(loop, 'shutdown_asyncgens'):  # Python 3.6+
                loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            asyncio.set_event_loop(None)
            loop.close()