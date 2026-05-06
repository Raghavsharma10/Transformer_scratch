def _add_modifiers(self, sql, blueprint, column):
        """
        Add the column modifiers to the deifinition
        """
        for modifier in self._modifiers:
            method = '_modify_%s' % modifier

            if hasattr(self, method):
                sql += getattr(self, method)(blueprint, column)

        return sql