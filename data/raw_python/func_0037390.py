def destroy(self):
        """Undo the effects of initdb.

        Destroy all evidence of this DBMS, including its backing files.
        """
        self.stop()
        if self.base_pathname is not None:
            self._robust_remove(self.base_pathname)