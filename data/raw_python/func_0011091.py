def get_relaxation(self, A_configuration, B_configuration, I):
        """Get the sparse SDP relaxation of a Bell inequality.

        :param A_configuration: The definition of measurements of Alice.
        :type A_configuration: list of list of int.
        :param B_configuration: The definition of measurements of Bob.
        :type B_configuration: list of list of int.
        :param I: The matrix describing the Bell inequality in the
                  Collins-Gisin picture.
        :type I: list of list of int.
        """
        coefficients = collinsgisin_to_faacets(I)
        M, ncIndices = get_faacets_moment_matrix(A_configuration,
                                                 B_configuration, coefficients)
        self.n_vars = M.max() - 1
        bs = len(M)  # The block size
        self.block_struct = [bs]
        self.F = lil_matrix((bs**2, self.n_vars + 1))
        # Constructing the internal representation of the constraint matrices
        # See Section 2.1 in the SDPA manual and also Yalmip's internal
        # representation
        for i in range(bs):
            for j in range(i, bs):
                if M[i, j] != 0:
                    self.F[i*bs+j, abs(M[i, j])-1] = copysign(1, M[i, j])
        self.obj_facvar = [0 for _ in range(self.n_vars)]
        for i in range(1, len(ncIndices)):
            self.obj_facvar[abs(ncIndices[i])-2] += \
                copysign(1, ncIndices[i])*coefficients[i]