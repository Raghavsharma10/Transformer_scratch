def is_duplicate_of(self, other):
        """Check if spectrum is duplicate of another."""
        if super(Spectrum, self).is_duplicate_of(other):
            return True
        row_matches = 0
        for ri, row in enumerate(self.get(self._KEYS.DATA, [])):
            lambda1, flux1 = tuple(row[0:2])
            if (self._KEYS.DATA not in other or
                    ri > len(other[self._KEYS.DATA])):
                break
            lambda2, flux2 = tuple(other[self._KEYS.DATA][ri][0:2])
            minlambdalen = min(len(lambda1), len(lambda2))
            minfluxlen = min(len(flux1), len(flux2))
            if (lambda1[:minlambdalen + 1] == lambda2[:minlambdalen + 1] and
                    flux1[:minfluxlen + 1] == flux2[:minfluxlen + 1] and
                    float(flux1[:minfluxlen + 1]) != 0.0):
                row_matches += 1
            # Five row matches should be enough to be sure spectrum is a dupe.
            if row_matches >= 5:
                return True
            # Matches need to happen in the first 10 rows.
            if ri >= 10:
                break
        return False