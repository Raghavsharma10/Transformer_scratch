def cancel_on_closing(func):
    """
    Automatically cancels a function or coroutine when the defining instance
    gets closed.

    :param func: The function to cancel on closing.
    :returns: A decorated function or coroutine.
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        return await self.await_until_closing(func(self, *args, **kwargs))

    return wrapper