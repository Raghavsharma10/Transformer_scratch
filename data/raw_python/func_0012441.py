def _by_columns(self, columns):
        """
        Allow select.group and select.order accepting string and list
        """
        return columns if self.isstr(columns) else self._backtick_columns(columns)