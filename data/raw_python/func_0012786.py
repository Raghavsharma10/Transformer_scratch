def savecopy(self, filename, lineendings='default', encoding='latin-1'):
        """Save a copy of the file with the filename passed.

        Parameters
        ----------
        filename : str
            Filepath to save the file.

        lineendings : str, optional
            Line endings to use in the saved file. Options are 'default',
            'windows' and 'unix' the default is 'default' which uses the line
            endings for the current system.

        encoding : str, optional
            Encoding to use for the saved file. The default is 'latin-1' which
            is compatible with the EnergyPlus IDFEditor.

        """
        self.save(filename, lineendings, encoding)