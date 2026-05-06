def find_solution_ranks(self, xmat=None, baselevel=0):
        """Helper function to detect rank loop in the solution matrix.

        :param sdpRelaxation: The SDP relaxation.
        :type sdpRelaxation: :class:`ncpol2sdpa.SdpRelaxation`.
        :param x_mat: Optional parameter providing the primal solution of the
                      moment matrix. If not provided, the solution is extracted
                      from the sdpRelaxation object.
        :type x_mat: :class:`numpy.array`.
        :param base_level: Optional parameter for specifying the lower level
                           relaxation for which the rank loop should be tested
                           against.
        :type base_level: int.
        :returns: list of int -- the ranks of the solution matrix with in the
                  order of increasing degree.
        """
        return find_solution_ranks(self, xmat=xmat, baselevel=baselevel)