def open(self, name, mode="r", pwd=None):
        """Return file-like object for 'name'."""
        if mode not in ("r", "U", "rU"):
            raise RuntimeError('open() requires mode "r", "U", or "rU"')
        if not self.fp:
            raise RuntimeError(
                "Attempt to read ZIP archive that was already closed"
            )

        # Only open a new file for instances where we were not
        # given a file object in the constructor
        if self._filePassed:
            zef_file = self.fp
        else:
            zef_file = open(self.filename, 'rb')

        # Make sure we have an info object
        if isinstance(name, ZipInfo):
            # 'name' is already an info object
            zinfo = name
        else:
            # Get info object for name
            zinfo = self.getinfo(name)

        zef_file.seek(zinfo.header_offset, 0)

        # Skip the file header:
        fheader = zef_file.read(sizeFileHeader)
        if fheader[0:4] != stringFileHeader:
            raise BadZipfile("Bad magic number for file header")

        fheader = struct.unpack(structFileHeader, fheader)
        fname = zef_file.read(fheader[_FH_FILENAME_LENGTH])
        if fheader[_FH_EXTRA_FIELD_LENGTH]:
            zef_file.read(fheader[_FH_EXTRA_FIELD_LENGTH])

        if fname != zinfo.orig_filename.encode('utf-8'):
            raise BadZipfile(
                'File name in directory "%s" and header "%s" differ.' % (
                    zinfo.orig_filename, fname)
            )

        # check for encrypted flag & handle password
        is_encrypted = zinfo.flag_bits & 0x1
        zd = None
        if is_encrypted:
            if not pwd:
                pwd = self.pwd
            if not pwd:
                raise RuntimeError("File %s is encrypted, " \
                      "password required for extraction" % name)

            zd = _ZipDecrypter(pwd)
            # The first 12 bytes in the cypher stream is an encryption header
            #  used to strengthen the algorithm. The first 11 bytes are
            #  completely random, while the 12th contains the MSB of the CRC,
            #  or the MSB of the file time depending on the header type
            #  and is used to check the correctness of the password.
            bytes = zef_file.read(12)
            h = map(zd, bytes[0:12])
            if zinfo.flag_bits & 0x8:
                # compare against the file type from extended local headers
                check_byte = (zinfo._raw_time >> 8) & 0xff
            else:
                # compare against the CRC otherwise
                check_byte = (zinfo.CRC >> 24) & 0xff
            if ord(h[11]) != check_byte:
                raise RuntimeError("Bad password for file", name)

        return  ZipExtFile(zef_file, mode, zinfo, zd)