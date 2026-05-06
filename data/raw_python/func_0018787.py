def set_emission_scenario_setup(self, scenario, config_dict):
        """Set the emissions flags correctly.

        Parameters
        ----------
        scenario : :obj:`pymagicc.io.MAGICCData`
            Scenario to run.

        config_dict : dict
            Dictionary with current input configurations which is to be validated and
            updated where necessary.

        Returns
        -------
        dict
            Updated configuration
        """
        self.write(scenario, self._scen_file_name)
        # can be lazy in this line as fix backwards key handles errors for us
        config_dict["file_emissionscenario"] = self._scen_file_name
        config_dict = self._fix_any_backwards_emissions_scen_key_in_config(config_dict)

        return config_dict