def read_parameters(self):
        """
        Read a parameters.out file

        Returns
        -------
        dict
            A dictionary containing all the configuration used by MAGICC
        """
        param_fname = join(self.out_dir, "PARAMETERS.OUT")

        if not exists(param_fname):
            raise FileNotFoundError("No PARAMETERS.OUT found")

        with open(param_fname) as nml_file:
            parameters = dict(f90nml.read(nml_file))
            for group in ["nml_years", "nml_allcfgs", "nml_outputcfgs"]:
                parameters[group] = dict(parameters[group])
                for k, v in parameters[group].items():
                    parameters[group][k] = _clean_value(v)
                parameters[group.replace("nml_", "")] = parameters.pop(group)
            self.config = parameters
        return parameters