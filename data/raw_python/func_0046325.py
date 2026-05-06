def _stream_out(self, outfile, append=False):
        '''
        Internal. Writes all stdout into outfile.
        :param outfile: Filename or file-like object for writing.
        :param append: Opens filename with append.
        :return: This command's returncode.
        '''
        if type(outfile) in (str, unicode):
            outfile = os.path.expanduser(os.path.expandvars(outfile))
            outfile = open(outfile, 'a' if append else 'w')
        self._run(outfile)
        self._pop.wait()