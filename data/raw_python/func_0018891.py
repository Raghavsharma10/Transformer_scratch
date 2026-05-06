def load_file(self, sequence):
        """Load data from an "external" data file an pass it to
        the given |IOSequence|."""
        try:
            if sequence.filetype_ext == 'npy':
                sequence.series = sequence.adjust_series(
                    *self._load_npy(sequence))
            elif sequence.filetype_ext == 'asc':
                sequence.series = sequence.adjust_series(
                    *self._load_asc(sequence))
            elif sequence.filetype_ext == 'nc':
                self._load_nc(sequence)
        except BaseException:
            objecttools.augment_excmessage(
                'While trying to load the external data of sequence %s'
                % objecttools.devicephrase(sequence))