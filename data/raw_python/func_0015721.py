def compile(self, **kwargs):
        """Execute the python code and returns the global dict.
        kwargs can contain extra dependencies that get only used
        at compile time.
        """

        code = compile(str(self), "<string>", "exec")
        global_dict = dict(self._deps)
        global_dict.update(kwargs)
        _compat.exec_(code, global_dict)
        return global_dict