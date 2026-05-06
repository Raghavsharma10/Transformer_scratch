def connect(
        database: str,
        loop: asyncio.BaseEventLoop = None,
        executor: concurrent.futures.Executor = None,
        timeout: int = 5,
        echo: bool = False,
        isolation_level: str = '',
        check_same_thread: bool = False,
        **kwargs: dict
):
    """
    把async方法执行后的对象创建为async上下文模式
    """
    coro = _connect(
        database,
        loop=loop,
        executor=executor,
        timeout=timeout,
        echo=echo,
        isolation_level=isolation_level,
        check_same_thread=check_same_thread,
        **kwargs
    )
    return _ContextManager(coro)