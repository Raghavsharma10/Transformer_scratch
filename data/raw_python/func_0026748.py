def is_erroneous(self, field, sources):
        """Check if attribute has been marked as being erroneous."""
        if self._KEYS.ERRORS in self:
            my_errors = self[self._KEYS.ERRORS]
            for alias in sources.split(','):
                source = self.get_source_by_alias(alias)
                bib_err_values = [
                    err[ERROR.VALUE] for err in my_errors
                    if err[ERROR.KIND] == SOURCE.BIBCODE and
                    err[ERROR.EXTRA] == field
                ]
                if (SOURCE.BIBCODE in source and
                        source[SOURCE.BIBCODE] in bib_err_values):
                    return True

                name_err_values = [
                    err[ERROR.VALUE] for err in my_errors
                    if err[ERROR.KIND] == SOURCE.NAME and err[ERROR.EXTRA] ==
                    field
                ]
                if (SOURCE.NAME in source and
                        source[SOURCE.NAME] in name_err_values):
                    return True

        return False