def invoke(*args, **kwargs):
        """Invokes a command callback in exactly the way it expects.  There
        are two ways to invoke this method:

        1.  the first argument can be a callback and all other arguments and
            keyword arguments are forwarded directly to the function.
        2.  the first argument is a click command object.  In that case all
            arguments are forwarded as well but proper click parameters
            (options and click arguments) must be keyword arguments and Click
            will fill in defaults.

        Note that before Click 3.2 keyword arguments were not properly filled
        in against the intention of this code and no context was created.  For
        more information about this change and why it was done in a bugfix
        release see :ref:`upgrade-to-3.2`.
        """
        self, callback = args[:2]
        ctx = self

        # This is just to improve the error message in cases where old
        # code incorrectly invoked this method.  This will eventually be
        # removed.
        injected_arguments = False

        # It's also possible to invoke another command which might or
        # might not have a callback.  In that case we also fill
        # in defaults and make a new context for this command.
        if isinstance(callback, Command):
            other_cmd = callback
            callback = other_cmd.callback
            ctx = Context(other_cmd, info_name=other_cmd.name, parent=self)
            if callback is None:
                raise TypeError('The given command does not have a '
                                'callback that can be invoked.')

            for param in other_cmd.params:
                if param.name not in kwargs and param.expose_value:
                    kwargs[param.name] = param.get_default(ctx)
                    injected_arguments = True

        args = args[2:]
        if getattr(callback, '__click_pass_context__', False):
            args = (ctx,) + args
        with augment_usage_errors(self):
            try:
                with ctx:
                    return callback(*args, **kwargs)
            except TypeError as e:
                if not injected_arguments:
                    raise
                if 'got multiple values for' in str(e):
                    raise RuntimeError(
                        'You called .invoke() on the context with a command '
                        'but provided parameters as positional arguments.  '
                        'This is not supported but sometimes worked by chance '
                        'in older versions of Click.  To fix this see '
                        'http://click.pocoo.org/upgrading/#upgrading-to-3.2')
                raise