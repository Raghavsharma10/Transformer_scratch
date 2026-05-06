def open(self, file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        """Downloads file from WebDAV server and saves it temprorary, then opens it for further manipulations.
        Has the same interface as built-in open()

        :param file: the path to remote file for opening.
        """
        urn = Urn(file)
        urn_path = urn.path()

        remote_file_exists = self.check(urn_path)

        if not remote_file_exists:
            if 'r' in mode:
                raise RemoteResourceNotFound(urn_path)
        elif self.is_dir(urn_path):
            raise OptionNotValid(name='file', value=file)

        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = f'{temp_dir}{os.path.sep}{file}'

            if remote_file_exists:
                self.download_file(file, local_path)
            else:
                if ('w' in mode or 'a' in mode or 'x' in mode) and os.path.sep in local_path:
                    os.makedirs(local_path.rsplit(os.path.sep, 1)[0], exist_ok=True)

            with open(file=local_path, mode=mode, buffering=buffering, encoding=encoding, errors=errors,
                      newline=newline, closefd=closefd, opener=opener) as f:
                yield f

            if 'w' in mode or 'a' in mode or 'x' in mode:
                self.upload_file(file, local_path)