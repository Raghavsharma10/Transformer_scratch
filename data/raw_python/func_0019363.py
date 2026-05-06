def get_index(self, name_subdevice) -> int:
        """Item access to the wrapped |dict| object with a specialized
        error message."""
        try:
            return self.dict_[name_subdevice]
        except KeyError:
            raise OSError(
                'No data for sequence `%s` and (sub)device `%s` '
                'in NetCDF file `%s` available.'
                % (self.name_sequence,
                   name_subdevice,
                   self.name_ncfile))