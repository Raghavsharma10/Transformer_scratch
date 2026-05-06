def save(self, filename=None, lineendings='default', encoding='latin-1'):
        """
        Save the IDF as a text file with the optional filename passed, or with
        the current idfname of the IDF.

        Parameters
        ----------
        filename : str, optional
            Filepath to save the file. If None then use the IDF.idfname
            parameter. Also accepts a file handle.

        lineendings : str, optional
            Line endings to use in the saved file. Options are 'default',
            'windows' and 'unix' the default is 'default' which uses the line
            endings for the current system.

        encoding : str, optional
            Encoding to use for the saved file. The default is 'latin-1' which
            is compatible with the EnergyPlus IDFEditor.

        """
        if filename is None:
            filename = self.idfname
        s = self.idfstr()
        if lineendings == 'default':
            system = platform.system()
            s = '!- {} Line endings \n'.format(system) + s
            slines = s.splitlines()
            s = os.linesep.join(slines)
        elif lineendings == 'windows':
            s = '!- Windows Line endings \n' + s
            slines = s.splitlines()
            s = '\r\n'.join(slines)
        elif lineendings == 'unix':
            s = '!- Unix Line endings \n' + s
            slines = s.splitlines()
            s = '\n'.join(slines)

        s = s.encode(encoding)
        try:
            with open(filename, 'wb') as idf_out:
                idf_out.write(s)
        except TypeError:  # in the case that filename is a file handle
            try:
                filename.write(s)
            except TypeError:
                filename.write(s.decode(encoding))