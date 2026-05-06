def save_file(self, sequence, array=None):
        """Write the date stored in |IOSequence.series| of the given
        |IOSequence| into an "external" data file. """
        if array is None:
            array = sequence.aggregate_series()
        try:
            if sequence.filetype_ext == 'nc':
                self._save_nc(sequence, array)
            else:
                filepath = sequence.filepath_ext
                if ((array is not None) and
                        (array.info['type'] != 'unmodified')):
                    filepath = (f'{filepath[:-4]}_{array.info["type"]}'
                                f'{filepath[-4:]}')
                if not sequence.overwrite_ext and os.path.exists(filepath):
                    raise OSError(
                        f'Sequence {objecttools.devicephrase(sequence)} '
                        f'is not allowed to overwrite the existing file '
                        f'`{sequence.filepath_ext}`.')
                if sequence.filetype_ext == 'npy':
                    self._save_npy(array, filepath)
                elif sequence.filetype_ext == 'asc':
                    self._save_asc(array, filepath)
        except BaseException:
            objecttools.augment_excmessage(
                'While trying to save the external data of sequence %s'
                % objecttools.devicephrase(sequence))