def _build_late_dispatcher(func_name):
        """Return a function that calls method 'func_name' on objects.

        This is useful for building late-bound dynamic dispatch.

        Arguments:
            func_name: The name of the instance method that should be called.

        Returns:
            A function that takes an 'obj' parameter, followed by *args and
            returns the result of calling the instance method with the same
            name as the contents of 'func_name' on the 'obj' object with the
            arguments from *args.
        """
        def _late_dynamic_dispatcher(obj, *args):
            method = getattr(obj, func_name, None)
            if not callable(method):
                raise NotImplementedError(
                    "Instance method %r is not implemented by %r." % (
                        func_name, obj))

            return method(*args)

        return _late_dynamic_dispatcher