def readall(cls, member, size):
        '''
        Parse variable metadata for a XPORT file member.
        '''
        fp = member.library.fp
        LINE = member.library.LINE

        n = cls.header_match(fp.read(LINE))
        namestrs = [fp.read(size) for i in range(n)]

        # Each namestr field is 140 bytes long, but the fields are
        # streamed together and broken in 80-byte pieces. If the last
        # byte of the last namestr field does not fall in the last byte
        # of the 80-byte record, the record is padded with ASCII blanks
        # to 80 bytes.

        remainder = n * size % LINE
        if remainder:
            padding = 80 - remainder
            fp.read(padding)

        info = [cls.unpack(s) for s in namestrs]
        for d in info:
            d['format'] = Format(**d['format'])
            d['iformat'] = InputFormat(**d['iformat'])
        return [Variable(**d) for d in info]