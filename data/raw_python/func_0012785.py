def saveas(self, filename, lineendings='default', encoding='latin-1'):
        """ Save the IDF as a text file with the filename passed.

        Parameters
        ----------
        filename : str
            Filepath to to set the idfname attribute to and save the file as.

        lineendings : str, optional
            Line endings to use in the saved file. Options are 'default',
            'windows' and 'unix' the default is 'default' which uses the line
            endings for the current system.

        encoding : str, optional
            Encoding to use for the saved file. The default is 'latin-1' which
            is compatible with the EnergyPlus IDFEditor.

        """
        self.idfname = filename
        self.save(filename, lineendings, encoding)