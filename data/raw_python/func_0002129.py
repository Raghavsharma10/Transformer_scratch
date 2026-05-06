def open_store_variable(self, name, var):
        """Turn CDMRemote variable into something like a numpy.ndarray."""
        data = indexing.LazilyOuterIndexedArray(CDMArrayWrapper(name, self))
        return Variable(var.dimensions, data, {a: getattr(var, a) for a in var.ncattrs()})