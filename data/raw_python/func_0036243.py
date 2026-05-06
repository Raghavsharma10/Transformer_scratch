def download(self,
                 files=None,
                 destination=None,
                 overwrite=False,
                 callback=None):

        """Download file or files.

        :param files: file or files to download
        :param destination: destination path (defaults to users home directory)
        :param overwrite: replace existing files?
        :param callback: callback function that will receive total file size
         and written bytes as arguments
        :type files: ``list`` of ``dict`` with file data from filemail
        :type destination: ``str`` or ``unicode``
        :type overwrite: ``bool``
        :type callback: ``func``
        """

        if files is None:
            files = self.files

        elif not isinstance(files, list):
            files = [files]

        if destination is None:
            destination = os.path.expanduser('~')

        for f in files:
            if not isinstance(f, dict):
                raise FMBaseError('File must be a <dict> with file data')

            self._download(f, destination, overwrite, callback)