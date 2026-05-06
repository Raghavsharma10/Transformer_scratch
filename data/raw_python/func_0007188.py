def _open(self):
        """
        Open the file; get sheets
        :return:
        """
        if not hasattr(self, '_file'):
            self._file = self.gc.open(self.name)
            self.sheet_names = self._file.worksheets()