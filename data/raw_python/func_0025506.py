def write_end_of_directory(fp, dir_size, dir_offset, count):
    """
        Write zip file end of directory header at the current file position

        :param fp: the file point to which to write the header
        :param dir_size: the total size of the directory
        :param dir_offset: the start of the first directory header
        :param count: the count of files
    """
    fp.write(struct.pack('I', 0x06054b50))  # central directory header
    fp.write(struct.pack('H', 0))           # disk number
    fp.write(struct.pack('H', 0))           # disk number
    fp.write(struct.pack('H', count))       # number of files
    fp.write(struct.pack('H', count))       # number of files
    fp.write(struct.pack('I', dir_size))    # central directory size
    fp.write(struct.pack('I', dir_offset))  # central directory offset
    fp.write(struct.pack('H', 0))