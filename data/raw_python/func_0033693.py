def __get_method_abbrev(self):
        """Abbreviated form of clustering method parameter.

        Used to guess output filenames for MOTHUR.
        """
        abbrevs = {
            'furthest': 'fn',
            'nearest': 'nn',
            'average': 'an',
        }
        if self.Parameters['method'].isOn():
            method = self.Parameters['method'].Value
        else:
            method = self.Parameters['method'].Default
        return abbrevs[method]