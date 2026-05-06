def _readfile(cls, filename):
        """ Reads a file a utf-8 file,
            and retuns character tokens.

            :param filename: Name of file to be read.
        """
        f = codecs.open(filename, encoding='utf-8')
        filedata = f.read()
        f.close()
        tokenz = LM.tokenize(filedata, mode='c')
        #print tokenz
        return tokenz