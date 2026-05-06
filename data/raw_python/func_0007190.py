def _import(self):
        """
        Makes imports
        :return:
        """
        import os.path
        import gspread
        self.path = os.path
        self.gspread = gspread
        self._login()