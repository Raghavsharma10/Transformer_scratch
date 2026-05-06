def set_years(self, startyear=1765, endyear=2100):
        """
        Set the start and end dates of the simulations.

        Parameters
        ----------
        startyear : int
            Start year of the simulation

        endyear : int
            End year of the simulation

        Returns
        -------
        dict
            The contents of the namelist
        """
        # TODO: test altering stepsperyear, I think 1, 2 and 24 should all work
        return self.set_config(
            "MAGCFG_NMLYEARS.CFG",
            "nml_years",
            endyear=endyear,
            startyear=startyear,
            stepsperyear=12,
        )