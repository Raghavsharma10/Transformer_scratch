def function_invocation_proxy(fn, proxy_args, proxy_kwargs):
        """execute the fuction if it is one, else evaluate the fn as a boolean
        and return that value.

        Sometimes rather than providing a predicate, we just give the value of
        True.  This is shorthand for writing a predicate that always returns
        true."""
        try:
            return fn(*proxy_args, **proxy_kwargs)
        except TypeError:
            return bool(fn)