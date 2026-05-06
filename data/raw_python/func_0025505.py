def write_directory_data(fp, offset, name_bytes, data_len, crc32, dt):
    """
        Write a zip fie directory entry at the current file position

        :param fp: the file point to which to write the header
        :param offset: the offset of the associated local file header
        :param name: the name of the file
        :param data_len: the length of data that will be written to the archive
        :param crc32: the crc32 of the data to be written
        :param dt: the datetime to write to the archive
    """
    fp.write(struct.pack('I', 0x02014b50))  # central directory header
    fp.write(struct.pack('H', 10))          # made by version (default)
    fp.write(struct.pack('H', 10))          # extract version (default)
    fp.write(struct.pack('H', 0))           # general purpose bits
    fp.write(struct.pack('H', 0))           # compression method
    msdos_date = int(dt.year - 1980) << 9 | int(dt.month) << 5 | int(dt.day)
    msdos_time = int(dt.hour) << 11 | int(dt.minute) << 5 | int(dt.second)
    fp.write(struct.pack('H', msdos_time))  # extract version (default)
    fp.write(struct.pack('H', msdos_date))  # extract version (default)
    fp.write(struct.pack('I', crc32))       # crc32
    fp.write(struct.pack('I', data_len))    # compressed length
    fp.write(struct.pack('I', data_len))    # uncompressed length
    fp.write(struct.pack('H', len(name_bytes)))   # name length
    fp.write(struct.pack('H', 0))           # extra length
    fp.write(struct.pack('H', 0))           # comments length
    fp.write(struct.pack('H', 0))           # disk number
    fp.write(struct.pack('H', 0))           # internal file attributes
    fp.write(struct.pack('I', 0))           # external file attributes
    fp.write(struct.pack('I', offset))      # relative offset of file header
    fp.write(name_bytes)