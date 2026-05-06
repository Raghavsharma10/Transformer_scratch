def add_spectrum(self, compare_to_existing=True, **kwargs):
        """Add a `Spectrum` instance to this entry."""
        spec_key = self._KEYS.SPECTRA
        # Make sure that a source is given, and is valid (nor erroneous)
        source = self._check_cat_dict_source(Spectrum, spec_key, **kwargs)
        if source is None:
            return None

        # Try to create a new instance of `Spectrum`
        new_spectrum = self._init_cat_dict(Spectrum, spec_key, **kwargs)
        if new_spectrum is None:
            return None

        is_dupe = False
        for item in self.get(spec_key, []):
            # Only the `filename` should be compared for duplicates. If a
            # duplicate is found, that means the previous `exclude` array
            # should be saved to the new object, and the old deleted
            if new_spectrum.is_duplicate_of(item):
                if SPECTRUM.EXCLUDE in new_spectrum:
                    item[SPECTRUM.EXCLUDE] = new_spectrum[SPECTRUM.EXCLUDE]
                elif SPECTRUM.EXCLUDE in item:
                    item.update(new_spectrum)
                is_dupe = True
                break

        if not is_dupe:
            self.setdefault(spec_key, []).append(new_spectrum)
        return