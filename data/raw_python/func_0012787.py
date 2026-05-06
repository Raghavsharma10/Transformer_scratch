def run(self, **kwargs):
        """
        Run an IDF file with a given EnergyPlus weather file. This is a
        wrapper for the EnergyPlus command line interface.

        Parameters
        ----------
        **kwargs
            See eppy.runner.functions.run()

        """
        # write the IDF to the current directory
        self.saveas('in.idf')
        # if `idd` is not passed explicitly, use the IDF.iddname
        idd = kwargs.pop('idd', self.iddname)
        epw = kwargs.pop('weather', self.epw)
        try:
            run(self, weather=epw, idd=idd, **kwargs)
        finally:
            os.remove('in.idf')