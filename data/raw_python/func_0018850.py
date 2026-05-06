def get_submask(self, *args, **kwargs) -> masktools.CustomMask:
        """Get a sub-mask of the mask handled by the actual |Variable| object
        based on the given arguments.

        See the documentation on method |Variable.average_values| for
        further information.
        """
        if args or kwargs:
            masks = self.availablemasks
            mask = masktools.CustomMask(numpy.full(self.shape, False))
            for arg in args:
                mask = mask + self._prepare_mask(arg, masks)
            for key, value in kwargs.items():
                mask = mask + self._prepare_mask(key, masks, **value)
            if mask not in self.mask:
                raise ValueError(
                    f'Based on the arguments `{args}` and `{kwargs}` '
                    f'the mask `{repr(mask)}` has been determined, '
                    f'which is not a submask of `{repr(self.mask)}`.')
        else:
            mask = self.mask
        return mask