def _create_scope(self):
        """TODO: Docstring for _create_scope.
        :returns: TODO

        """
        res = Scope(self._log)

        for func_name,native_func in six.iteritems(self._natives):
            res.add_local(func_name, native_func)

        return res