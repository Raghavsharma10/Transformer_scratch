def parse_zip(fp):
    """
        Parse the zip file headers at fp

        :param fp: the file pointer from which to parse the zip file
        :return: A tuple of local files, directory headers, and end of central directory

        The local files are dictionary where the keys are the local file offset and the
        values are each a tuple consisting of the name, data position, data length, and crc32.

        The directory headers are a dictionary where the keys are the names of the files
        and the values are a tuple consisting of the directory header position, and the
        associated local file position.

        The end of central directory is a tuple consisting of the location of the end of
        central directory header and the location of the first directory header.

        This method will seek to location 0 of fp and leave fp at end of file.
    """
    local_files = {}
    dir_files = {}
    eocd = None
    fp.seek(0)
    while True:
        pos = fp.tell()
        signature = struct.unpack('I', fp.read(4))[0]
        if signature == 0x04034b50:
            fp.seek(pos + 14)
            crc32 = struct.unpack('I', fp.read(4))[0]
            fp.seek(pos + 18)
            data_len = struct.unpack('I', fp.read(4))[0]
            fp.seek(pos + 26)
            name_len = struct.unpack('H', fp.read(2))[0]
            extra_len = struct.unpack('H', fp.read(2))[0]
            name_bytes = fp.read(name_len)
            fp.seek(extra_len, os.SEEK_CUR)
            data_pos = fp.tell()
            fp.seek(data_len, os.SEEK_CUR)
            local_files[pos] = (name_bytes, data_pos, data_len, crc32)
        elif signature == 0x02014b50:
            fp.seek(pos + 28)
            name_len = struct.unpack('H', fp.read(2))[0]
            extra_len = struct.unpack('H', fp.read(2))[0]
            comment_len = struct.unpack('H', fp.read(2))[0]
            fp.seek(pos + 42)
            pos2 = struct.unpack('I', fp.read(4))[0]
            name_bytes = fp.read(name_len)
            fp.seek(pos + 46 + name_len + extra_len + comment_len)
            dir_files[name_bytes] = (pos, pos2)
        elif signature == 0x06054b50:
            fp.seek(pos + 16)
            pos2 = struct.unpack('I', fp.read(4))[0]
            eocd = (pos, pos2)
            break
        else:
            raise IOError()
    return local_files, dir_files, eocd