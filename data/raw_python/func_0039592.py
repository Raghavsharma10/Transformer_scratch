def async_atomic(on_exception=None, raise_exception=True, **kwargs):
    '''
    first argument will be a conn object
    :param func:
    :return:
    '''
    if not raise_exception and not on_exception:
        async def default_on_exception(exc):
            resp_dict = {}
            resp_dict['status'] = type(exc)
            resp_dict['message'] = str(exc)
            return resp_dict

        on_exception = default_on_exception
    elif raise_exception and not on_exception:
        async def raise_exception(exp_args):
            raise exp_args

        on_exception = raise_exception

    _db_adapter = get_db_adapter()

    def decorator(func):
        @functools.wraps(func)
        async def wrapped(self, *args, **kwargs):
            conn = None
            for i in itertools.chain(args, kwargs.values()):
                if type(i) is Connection:
                    conn = i
                    break
            if not conn:
                pool = await _db_adapter.get_pool()
                async with pool.acquire() as conn:
                    try:
                        async with conn.transaction():
                            kwargs['conn'] = conn
                            return await func(self, *args, **kwargs)
                    except Exception as e:
                        return await on_exception(e)
            else:
                try:
                    async with conn.transaction():
                        kwargs['conn'] = conn
                        return await func(self, *args, **kwargs)
                except Exception as e:
                    return await on_exception(e)

        return wrapped

    return decorator