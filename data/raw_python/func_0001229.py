def get_filename(self, checksum):
        """
        :param checksum: checksum
        :return: filename no storage base part
        """
        filename = None
        for _filename, metadata in self._log.items():
            if metadata['checksum'] == checksum:
                filename = _filename
                break
        return filename