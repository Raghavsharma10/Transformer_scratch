def solve(self, solver=None, solverparameters=None):
        """Call a solver on the SDP relaxation. Upon successful solution, it
        returns the primal and dual objective values along with the solution
        matrices. It also sets these values in the `sdpRelaxation` object,
        along with some status information.

        :param sdpRelaxation: The SDP relaxation to be solved.
        :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.
        :param solver: The solver to be called, either `None`, "sdpa", "mosek",
                       "cvxpy", "scs", or "cvxopt". The default is `None`,
                       which triggers autodetect.
        :type solver: str.
        :param solverparameters: Parameters to be passed to the solver. Actual
                                 options depend on the solver:

                                 SDPA:

                                   - `"executable"`:
                                     Specify the executable for SDPA. E.g.,
                                     `"executable":"/usr/local/bin/sdpa"`, or
                                     `"executable":"sdpa_gmp"`
                                   - `"paramsfile"`: Specify the parameter file

                                 Mosek:
                                 Refer to the Mosek documentation. All
                                 arguments are passed on.

                                 Cvxopt:
                                 Refer to the PICOS documentation. All
                                 arguments are passed on.

                                 Cvxpy:
                                 Refer to the Cvxpy documentation. All
                                 arguments are passed on.

                                 SCS:
                                 Refer to the Cvxpy documentation. All
                                 arguments are passed on.

        :type solverparameters: dict of str.
        """
        if self.F is None:
            raise Exception("Relaxation is not generated yet. Call "
                            "'SdpRelaxation.get_relaxation' first")
        solve_sdp(self, solver, solverparameters)