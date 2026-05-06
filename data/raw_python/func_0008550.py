def pre_process(self, x0, params=()):
        """ Used internally for transformation of variables. """
        # Should be used by all methods matching "solve_*"
        if self.x_by_name and isinstance(x0, dict):
            x0 = [x0[k] for k in self.names]
        if self.par_by_name and isinstance(params, dict):
            params = [params[k] for k in self.param_names]
        for pre_processor in self.pre_processors:
            x0, params = pre_processor(x0, params)
        return x0, np.atleast_1d(params)