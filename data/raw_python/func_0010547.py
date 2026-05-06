def _calculate_osdb_hash(self):
        """
        Calculate OSDB (OpenSubtitleDataBase) hash of this VideoFile
        :return: hash as string
        """
        log.debug('_calculate_OSDB_hash() of "{path}" ...'.format(path=self._filepath))
        f = self._filepath.open(mode='rb')

        file_size = self.get_size()

        longlong_format = 'Q'  # unsigned long long little endian
        size_longlong = struct.calcsize(longlong_format)

        block_size = min(file_size, 64 << 10)  # 64kiB
        block_size = block_size & ~(size_longlong - 1)  # lower round on multiple of longlong

        nb_longlong = block_size // size_longlong
        fmt = '<{nbll}{member_format}'.format(
            nbll=nb_longlong,
            member_format=longlong_format)

        hash_int = file_size

        buffer = f.read(block_size)
        list_longlong = struct.unpack(fmt, buffer)
        hash_int += sum(list_longlong)

        f.seek(-block_size, os.SEEK_END)
        buffer = f.read(block_size)
        list_longlong = struct.unpack(fmt, buffer)
        hash_int += sum(list_longlong)

        f.close()
        hash_str = '{:016x}'.format(hash_int)[-16:]
        log.debug('hash("{}")={}'.format(self.get_filepath(), hash_str))
        return hash_str