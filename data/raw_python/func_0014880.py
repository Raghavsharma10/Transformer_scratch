def _set(self, name, value):
        r"""Directly set a variable `name` in matlab space to `value`.

        This should normally not be used in user code."""
        if isinstance(value, MlabObjectProxy):
            mlabraw.eval(self._session, "%s = %s;" % (name, value._name))
        else:
##             mlabraw.put(self._session, name, self._as_mlabable_type(value))
            mlabraw.put(self._session, name, value)