def description(self, argv0='manage.py', command=None):
        '''Description outputed to console'''
        command = command or self.__class__.__name__.lower()
        import inspect
        _help = u''
        _help += u'{}\n'.format(command)
        if self.__doc__:
            _help += self._fix_docstring(self.__doc__) +'\n'
        else:
            _help += u'{}\n'.format(command)

        funcs = self.get_funcs()
        funcs.sort(key=lambda x: six.get_function_code(x[1]).co_firstlineno)

        for attr, func in funcs:
            func = getattr(self, attr)
            comm = attr.replace('command_', '', 1)
            args = inspect.getargspec(func).args[1:]
            args = (' [' + '] ['.join(args) + ']') if args else ''

            _help += "\t{} {}:{}{}\n".format(
                            argv0, command, comm, args)

            if func.__doc__:
                _help += self._fix_docstring(func.__doc__, 2)
        return _help