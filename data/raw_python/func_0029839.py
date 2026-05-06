def remove(self):
        """ Removes file from filesystem. """
        from fs.errors import ResourceNotFoundError

        try:
            self._fs.remove(self.file_name)
        except ResourceNotFoundError:
            pass