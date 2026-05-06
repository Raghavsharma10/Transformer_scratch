def getTmpFilename(self, tmp_dir=None, prefix='tmp', suffix='.txt'):
        """Returns a temporary filename

        Similar interface to tempfile.mktmp()
        """
        # Override to change default constructor to str(). FilePath
        # objects muck up the Mothur script.
        return super(Mothur, self).getTmpFilename(
            tmp_dir=tmp_dir, prefix=prefix, suffix=suffix,
            result_constructor=str)