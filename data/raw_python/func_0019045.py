def new(cls, variable, **kwargs):
        """Return a new |IndexMask| object of the same shape as the
        parameter referenced by |property| |IndexMask.refindices|.
        Entries are only |True|, if the integer values of the
        respective entries of the referenced parameter are contained
        in the |IndexMask| class attribute tuple `RELEVANT_VALUES`.
        """
        indices = cls.get_refindices(variable)
        if numpy.min(getattr(indices, 'values', 0)) < 1:
            raise RuntimeError(
                f'The mask of parameter {objecttools.elementphrase(variable)} '
                f'cannot be determined, as long as parameter `{indices.name}` '
                f'is not prepared properly.')
        mask = numpy.full(indices.shape, False, dtype=bool)
        refvalues = indices.values
        for relvalue in cls.RELEVANT_VALUES:
            mask[refvalues == relvalue] = True
        return cls.array2mask(mask, **kwargs)