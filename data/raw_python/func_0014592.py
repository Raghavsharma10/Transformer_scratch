def download(self, file_name, save_as=None):
        """Download the specified file from the server.

        Arguments:
        file_name -- Name of file resource to save.
        save_as   -- Optional path name to write file to.  If not specified,
                     then file named by the last part of the resource path is
                     downloaded to current directory.

        Return: (save_path, bytes)
        save_path -- Path where downloaded file was saved.
        bytes     -- Bytes downloaded.
        """
        self._check_session()
        try:
            if save_as:
                save_as = os.path.normpath(save_as)
                save_dir = os.path.dirname(save_as)
                if save_dir:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    elif not os.path.isdir(save_dir):
                        raise RuntimeError(save_dir + " is not a directory")

            status, save_path, bytes = self._rest.download_file(
                'files', file_name, save_as, 'application/octet-stream')
        except resthttp.RestHttpError as e:
            raise RuntimeError('failed to download "%s": %s' % (file_name, e))
        return save_path, bytes