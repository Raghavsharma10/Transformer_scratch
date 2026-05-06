def create_pool(
        database,
        minsize=1,
        maxsize=10,
        echo=False,
        loop=None,
        **kwargs
):
    """
    创建支持上下文管理的pool
    """
    coro = _create_pool(
        database=database,
        minsize=minsize,
        maxsize=maxsize,
        echo=echo,
        loop=loop,
        **kwargs
    )
    return _PoolContextManager(coro)