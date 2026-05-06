def _get(self, name, remove=False):
        r"""Directly access a variable in matlab space.

        This should normally not be used by user code."""
        # FIXME should this really be needed in normal operation?
        if name in self._proxies: return self._proxies[name]
        varname = name
        vartype = self._var_type(varname)
        if vartype in self._mlabraw_can_convert:
            var = mlabraw.get(self._session, varname)
            if isinstance(var, ndarray):
                if self._flatten_row_vecs and numpy.shape(var)[0] == 1:
                    var.shape = var.shape[1:2]
                elif self._flatten_col_vecs and numpy.shape(var)[1] == 1:
                    var.shape = var.shape[0:1]
                if self._array_cast:
                    var = self._array_cast(var)
        else:
            var = None
            if self._dont_proxy.get(vartype):
                # manual conversions may fail (e.g. for multidimensional
                # cell arrays), in that case just fall back on proxying.
                try:
                    var = self._manually_convert(varname, vartype)
                except MlabConversionError: pass
            if var is None:
                # we can't convert this to a python object, so we just
                # create a proxy, and don't delete the real matlab
                # reference until the proxy is garbage collected
                var = self._make_proxy(varname)
        if remove:
            mlabraw.eval(self._session, "clear('%s');" % varname)
        return var