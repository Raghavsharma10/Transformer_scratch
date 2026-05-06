def restore(self):
        """ Reset the active WCS keywords to values stored in the
            backup keywords.
        """
        # If there are no backup keys, do nothing...
        if len(list(self.backup.keys())) == 0:
            return
        for key in self.backup.keys():
            if key != 'WCSCDATE':
                self.__dict__[self.wcstrans[key]] = self.orig_wcs[self.backup[key]]

        self.update()