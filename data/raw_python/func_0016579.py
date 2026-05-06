def sample_qubo(self, qubo, **params):
        """Sample from the specified QUBO.

        Args:
            qubo (dict of (int, int):float): Coefficients of a quadratic unconstrained binary
                optimization (QUBO) model.
            **params: Parameters for the sampling method, specified per solver.

        Returns:
            :obj:`Future`

        Examples:
            This example creates a client using the local system's default D-Wave Cloud Client
            configuration file, which is configured to access a D-Wave 2000Q QPU, submits
            a :term:`QUBO` problem (a Boolean NOT gate represented by a penalty model), and
            samples 5 times.

            >>> from dwave.cloud import Client
            >>> with Client.from_config() as client:  # doctest: +SKIP
            ...     solver = client.get_solver()
            ...     u, v = next(iter(solver.edges))
            ...     Q = {(u, u): -1, (u, v): 0, (v, u): 2, (v, v): -1}
            ...     computation = solver.sample_qubo(Q, num_reads=5)
            ...     for i in range(5):
            ...         print(computation.samples[i][u], computation.samples[i][v])
            ...
            ...
            (0, 1)
            (1, 0)
            (1, 0)
            (0, 1)
            (1, 0)

        """
        # In a QUBO the linear and quadratic terms in the objective are mixed into
        # a matrix. For the sake of encoding, we will separate them before calling `_sample`
        linear = {i1: v for (i1, i2), v in uniform_iterator(qubo) if i1 == i2}
        quadratic = {(i1, i2): v for (i1, i2), v in uniform_iterator(qubo) if i1 != i2}
        return self._sample('qubo', linear, quadratic, params)