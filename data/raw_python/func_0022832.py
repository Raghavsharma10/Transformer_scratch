def read(cls, fname):
        """ read(fname, fmt)

        This classmethod is the entry point for reading OBJ files.

        Parameters
        ----------
        fname : str
            The name of the file to read.
        fmt : str
            Can be "obj" or "gz" to specify the file format.
        """
        # Open file
        fmt = op.splitext(fname)[1].lower()
        assert fmt in ('.obj', '.gz')
        opener = open if fmt == '.obj' else gzip_open
        with opener(fname, 'rb') as f:
            try:
                reader = WavefrontReader(f)
                while True:
                    reader.readLine()
            except EOFError:
                pass

        # Done
        t0 = time.time()
        mesh = reader.finish()
        logger.debug('reading mesh took ' +
                     str(time.time() - t0) +
                     ' seconds')
        return mesh