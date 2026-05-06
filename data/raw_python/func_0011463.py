def _resolve_name(self, name):
        """TODO: Docstring for _resolve_names.

        :name: TODO
        :returns: TODO

        """
        res = [name]
        while True:
            orig_names = self._search("types", name)
            if orig_names is not None:
                name = orig_names[-1]
                # pop off the typedefd name
                res.pop()
                # add back on the original names
                res += orig_names
            else:
                break

        return res