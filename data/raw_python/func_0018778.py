def write(self, mdata, name):
        """Write an input file to disk

        Parameters
        ----------
        mdata : :obj:`pymagicc.io.MAGICCData`
            A MAGICCData instance with the data to write

        name : str
            The name of the file to write. The file will be written to the MAGICC
            instance's run directory i.e. ``self.run_dir``
        """
        mdata.write(join(self.run_dir, name), self.version)