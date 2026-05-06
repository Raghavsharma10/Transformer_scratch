def _get_file(self, path, prepend=False):
        '''Extracts a file from dap to a file-like object'''
        if prepend:
            path = os.path.join(self._dirname(), path)
        extracted = self._tar.extractfile(path)
        if extracted:
            return extracted
        raise DapFileError(('Could not read %s from %s, maybe it\'s a directory,' +
            'bad link or the dap file is corrupted') % (path, self.basename))