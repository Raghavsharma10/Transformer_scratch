def requires(*params):
    """
    Raise ValueError if any ``params`` are omitted from the decorated kwargs.

    None values are considered omissions.

    Example usage on an AWS() method:

        @requires('zone', 'security_groups')
        def my_aws_method(self, custom_args, **kwargs):
            # We'll only get here if 'kwargs' contained non-None values for
            # both 'zone' and 'security_groups'.
    """
    def requires(f, self, *args, **kwargs):
        missing = filter(lambda x: kwargs.get(x) is None, params)
        if missing:
            msgs = ", ".join([PARAMETERS[x]['msg'] for x in missing])
            raise ValueError("Missing the following parameters: %s" % msgs)
        return f(self, *args, **kwargs)
    return decorator(requires)