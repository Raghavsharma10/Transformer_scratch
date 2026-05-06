def load_from_stream(self, var):
        """Populate the Variable from an NCStream object."""
        dims = []
        for d in var.shape:
            dim = Dimension(None, d.name)
            dim.load_from_stream(d)
            dims.append(dim)

        self.dimensions = tuple(dim.name for dim in dims)
        self.shape = tuple(dim.size for dim in dims)
        self.ndim = len(var.shape)
        self._unpack_attrs(var.atts)

        data, dt, type_name = unpack_variable(var)
        if data is not None:
            data = data.reshape(self.shape)
        self._data = data
        self.dtype = dt
        self.datatype = type_name

        if hasattr(var, 'enumType') and var.enumType:
            self.datatype = var.enumType
            self._enum = True