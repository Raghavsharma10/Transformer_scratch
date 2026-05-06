def read_data(self, password=None):
        """
        Helper function which decrypts and decompresses the data if necessary
        and returns a tarfile.TarFile to interact with
        """
        fp = self.fp
        fp.seek(self.__data_start)

        if self.is_encrypted():
            fp = self._decrypt(fp, password=password)

        if self.compression == CompressionType.ZLIB:
            fp = self._decompress(fp)

        if self.stream:
            mode = 'r|*'
        else:
            mode = 'r:*'
        tar = tarfile.open(fileobj=fp, mode=mode)
        return tar