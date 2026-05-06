def open(cls, filename):
        """Opens a *.asar file and constructs a new :see AsarArchive instance.

        Args:
            filename (str):
                Path to the *.asar file to open for reading.

        Returns (AsarArchive):
            An insance of of the :AsarArchive class or None if reading failed.
        """

        asarfile = open(filename, 'rb')

        # uses google's pickle format, which prefixes each field
        # with its total length, the first field is a 32-bit unsigned
        # integer, thus 4 bytes, we know that, so we skip it
        asarfile.seek(4)

        header_size = struct.unpack('I', asarfile.read(4))
        if len(header_size) <= 0:
            raise IndexError()

        # substract 8 bytes from the header size, again because google's
        # pickle format uses some padding here
        header_size = header_size[0] - 8

        # read the actual header, which is a json string, again skip 8
        # bytes because of pickle padding
        asarfile.seek(asarfile.tell() + 8)
        header = asarfile.read(header_size).decode('utf-8')

        files = json.loads(header)
        return cls(filename, asarfile, files, asarfile.tell())