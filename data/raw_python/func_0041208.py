def _set_local_file_path(self):
        """
        Take from environment variable, create dirs and
        create file if doesn' exist.
        """

        self.FILE_LOCAL = self._transfer.get_env('FILE_LOCAL')

        if not self.FILE_LOCAL:
            filename = '{}_{}.{}'.format(str(self._transfer.prefix),
                                         str(self._transfer.namespace),
                                         str(self.file_extension))
            self.FILE_LOCAL = os.path.join(os.path.expanduser("~"), filename)

        dirs = os.path.dirname(self.FILE_LOCAL)
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        try:
            open(self.FILE_LOCAL, "rb+").close()
        except:
            open(self.FILE_LOCAL, "a").close()