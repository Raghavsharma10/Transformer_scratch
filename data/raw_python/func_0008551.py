def post_process(self, xout, params_out):
        """ Used internally for transformation of variables. """
        # Should be used by all methods matching "solve_*"
        for post_processor in self.post_processors:
            xout, params_out = post_processor(xout, params_out)
        return xout, params_out