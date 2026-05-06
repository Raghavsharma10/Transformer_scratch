def _merge_meta(left, right, result, clean=True):
        """Merge metadata from left and right onto results.

        This is used during class initialization.
        This should also be used by operators to merge metadata after
        creating a new instance but before returning it.
        Result's metadata is modified in-place.

        Parameters
        ----------
        left, right : number, `BaseSpectrum`, or `~astropy.modeling.models`
            Inputs of an operation.

        result : `BaseSpectrum`
            Output spectrum object.

        clean : bool
            Remove ``'header'`` and ``'expr'`` entries from inputs.

        """
        # Copies are returned because they need some clean-up below.
        left = BaseSpectrum._get_meta(left)
        right = BaseSpectrum._get_meta(right)

        # Remove these from going into result to avoid mess.
        #   header = FITS header metadata
        #   expr = ASTROLIB PYSYNPHOT expression
        if clean:
            for key in ('header', 'expr'):
                for d in (left, right):
                    if key in d:
                        del d[key]

        mid = metadata.merge(left, right, metadata_conflicts='silent')
        result.meta = metadata.merge(result.meta, mid,
                                     metadata_conflicts='silent')