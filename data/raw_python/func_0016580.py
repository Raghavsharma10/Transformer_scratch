def _sample(self, type_, linear, quadratic, params):
        """Internal method for both sample_ising and sample_qubo.

        Args:
            linear (list/dict): Linear terms of the model.
            quadratic (dict of (int, int):float): Quadratic terms of the model.
            **params: Parameters for the sampling method, specified per solver.

        Returns:
            :obj: `Future`
        """
        # Check the problem
        if not self.check_problem(linear, quadratic):
            raise ValueError("Problem graph incompatible with solver.")

        # Mix the new parameters with the default parameters
        combined_params = dict(self._params)
        combined_params.update(params)

        # Check the parameters before submitting
        for key in combined_params:
            if key not in self.parameters and not key.startswith('x_'):
                raise KeyError("{} is not a parameter of this solver.".format(key))

        # transform some of the parameters in-place
        self._format_params(type_, combined_params)

        body = json.dumps({
            'solver': self.id,
            'data': encode_bqm_as_qp(self, linear, quadratic),
            'type': type_,
            'params': combined_params
        })
        _LOGGER.trace("Encoded sample request: %s", body)

        future = Future(solver=self, id_=None, return_matrix=self.return_matrix,
                        submission_data=(type_, linear, quadratic, params))

        _LOGGER.debug("Submitting new problem to: %s", self.id)
        self.client._submit(body, future)
        return future