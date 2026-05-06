def _uses_aiohttp_session(func):
        """This is a decorator that creates an async with statement around a function, and makes sure that a _session argument is always passed.
        Only usable on async functions of course.
        The _session argument is (supposed to be) an aiohttp.ClientSession instance in all functions that this decorator has been used on.
        This is used to make sure that all session objects are properly entered and exited, or that they are passed into a function properly.
        This adds an session keyword argument to the method signature, and that session will be used as _session if it is not None."""

        # The function the decorator returns
        async def decorated_func(*args, session=None, **kwargs):
            if session is not None:
                # There is a session passed
                return await func(*args, _session=session, **kwargs)
            else:
                # The session argument wasn't passed, so we create our own
                async with aiohttp.ClientSession() as new_session:
                    return await func(*args, _session=new_session, **kwargs)

        # We return the decorated func
        return decorated_func