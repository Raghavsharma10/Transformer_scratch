def parse(self):
        """
        Parses a backup file header. Will be done automatically if
        used together with the 'with' statement
        """
        self.fp.seek(0)
        magic = self.fp.readline()
        assert magic == b'ANDROID BACKUP\n'
        self.version = int(self.fp.readline().strip())
        self.compression = CompressionType(int(self.fp.readline().strip()))
        self.encryption = EncryptionType(self.fp.readline().strip().decode())
        self.__data_start = self.fp.tell()