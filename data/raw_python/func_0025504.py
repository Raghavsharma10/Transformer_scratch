def write_local_file(fp, name_bytes, writer, dt):
    """
        Writes a zip file local file header structure at the current file position.

        Returns data_len, crc32 for the data.

        :param fp: the file point to which to write the header
        :param name: the name of the file
        :param writer: a function taking an fp parameter to do the writing, returns crc32
        :param dt: the datetime to write to the archive
    """
    fp.write(struct.pack('I', 0x04034b50))  # local file header
    fp.write(struct.pack('H', 10))          # extract version (default)
    fp.write(struct.pack('H', 0))           # general purpose bits
    fp.write(struct.pack('H', 0))           # compression method
    msdos_date = int(dt.year - 1980) << 9 | int(dt.month) << 5 | int(dt.day)
    msdos_time = int(dt.hour) << 11 | int(dt.minute) << 5 | int(dt.second)
    fp.write(struct.pack('H', msdos_time))  # extract version (default)
    fp.write(struct.pack('H', msdos_date))  # extract version (default)
    crc32_pos = fp.tell()
    fp.write(struct.pack('I', 0))           # crc32 placeholder
    data_len_pos = fp.tell()
    fp.write(struct.pack('I', 0))           # compressed length placeholder
    fp.write(struct.pack('I', 0))           # uncompressed length placeholder
    fp.write(struct.pack('H', len(name_bytes)))   # name length
    fp.write(struct.pack('H', 0))           # extra length
    fp.write(name_bytes)
    data_start_pos = fp.tell()
    crc32 = writer(fp)
    data_end_pos = fp.tell()
    data_len = data_end_pos - data_start_pos
    fp.seek(crc32_pos)
    fp.write(struct.pack('I', crc32))       # crc32
    fp.seek(data_len_pos)
    fp.write(struct.pack('I', data_len))    # compressed length placeholder
    fp.write(struct.pack('I', data_len))    # uncompressed length placeholder
    fp.seek(data_end_pos)
    return data_len, crc32