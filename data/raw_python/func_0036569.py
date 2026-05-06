def _get_abs_filepath(self, ifile):
        """ validate src or dst file path with self.config_file
        """
        assert ifile is not None
        ifile = ifile[7:] if ifile.startswith('file://') else ifile
        if ifile[0] != '/':
            basedir = os.path.abspath(os.path.dirname(self.config_file))
            ifile = os.path.join(basedir, ifile)
        return 'file://' + ifile