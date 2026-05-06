def _is_pickle_valid(self):
        """Logic to decide if the file should be processed or just needs to
        be loaded from its pickle data.
        """
        if not os.path.exists(self._pickle_file):
            return False
        else:
            file_mtime = os.path.getmtime(self.logfile)
            pickle_mtime = os.path.getmtime(self._pickle_file)
            if file_mtime > pickle_mtime:
                return False
        return True