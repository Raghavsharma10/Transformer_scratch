def local_filename(
            self,
            url=None,
            filename=None,
            decompress=False):
        """
        What local filename will we use within the cache directory
        for the given URL/filename/decompress options.
        """
        return common.build_local_filename(url, filename, decompress)