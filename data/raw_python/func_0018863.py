def compress_repr(self) -> Optional[str]:
        """Works as |Parameter.compress_repr|, but alternatively
        tries to compress by following an external classification.

        See the main documentation on class |ZipParameter| for
        further information.
        """
        string = super().compress_repr()
        if string is not None:
            return string
        results = []
        mask = self.mask
        refindices = mask.refindices.values
        for (key, value) in self.MODEL_CONSTANTS.items():
            if value in mask.RELEVANT_VALUES:
                unique = numpy.unique(self.values[refindices == value])
                unique = self.revert_timefactor(unique)
                length = len(unique)
                if length == 1:
                    results.append(
                        f'{key.lower()}={objecttools.repr_(unique[0])}')
                elif length > 1:
                    return None
        return ', '.join(sorted(results))