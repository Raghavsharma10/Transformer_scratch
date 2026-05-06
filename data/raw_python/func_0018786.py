def diagnose_tcr_ecs(self, **kwargs):
        """Diagnose TCR and ECS

        The transient climate response (TCR), is the global-mean temperature response
        at time at which atmopsheric |CO2| concentrations double in a scenario where
        atmospheric |CO2| concentrations are increased at 1% per year from
        pre-industrial levels.

        The equilibrium climate sensitivity (ECS), is the equilibrium global-mean
        temperature response to an instantaneous doubling of atmospheric |CO2|
        concentrations.

        As MAGICC has no hysteresis in its equilibrium response to radiative forcing,
        we can diagnose TCR and ECS with one experiment. However, please note that
        sometimes the run length won't be long enough to allow MAGICC's oceans to
        fully equilibrate and hence the ECS value might not be what you expect (it
        should match the value of ``core_climatesensitivity``).

        Parameters
        ----------
        **kwargs
            parameter values to use in the diagnosis e.g. ``core_climatesensitivity=4``

        Returns
        -------
        dict
            Dictionary with keys: "ecs" - the diagnosed ECS; "tcr" - the diagnosed
            TCR; "timeseries" - the relevant model input and output timeseries used in
            the experiment i.e. atmospheric |CO2| concentrations, total radiative
            forcing and global-mean surface temperature
        """
        if self.version == 7:
            raise NotImplementedError("MAGICC7 cannot yet diagnose ECS and TCR")
        self._diagnose_tcr_ecs_config_setup(**kwargs)
        timeseries = self.run(
            scenario=None,
            only=[
                "Atmospheric Concentrations|CO2",
                "Radiative Forcing",
                "Surface Temperature",
            ],
        )
        tcr, ecs = self._get_tcr_ecs_from_diagnosis_results(timeseries)
        return {"tcr": tcr, "ecs": ecs, "timeseries": timeseries}