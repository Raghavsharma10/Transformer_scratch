def load_commands(self, obj):
        """
        Load commands defined on an arbitrary object.

        All functions decorated with the :func:`subparse.command` decorator
        attached the specified object will be loaded. The object may
        be a dictionary, an arbitrary python object, or a dotted path.

        The dotted path may be absolute, or relative to the current package
        by specifying a leading '.' (e.g. ``'.commands'``).

        """
        if isinstance(obj, str):
            if obj.startswith('.') or obj.startswith(':'):
                package = caller_package()
                if obj in ['.', ':']:
                    obj = package.__name__
                else:
                    obj = package.__name__ + obj
            obj = pkg_resources.EntryPoint.parse('x=%s' % obj).resolve()
        command.discover_and_call(obj, self.command)