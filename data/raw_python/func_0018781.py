def set_config(
        self, filename="MAGTUNE_PYMAGICC.CFG", top_level_key="nml_allcfgs", **kwargs
    ):
        """
        Create a configuration file for MAGICC.

        Writes a fortran namelist in run_dir.

        Parameters
        ----------
        filename : str
            Name of configuration file to write

        top_level_key : str
            Name of namelist to be written in the
            configuration file

        kwargs
            Other parameters to pass to the configuration file. No
            validation on the parameters is performed.

        Returns
        -------
        dict
            The contents of the namelist which was written to file
        """
        kwargs = self._format_config(kwargs)

        fname = join(self.run_dir, filename)
        conf = {top_level_key: kwargs}
        f90nml.write(conf, fname, force=True)

        return conf