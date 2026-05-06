def __get_trace_facvar(self, polynomial):
        """Return dense vector representation of a polynomial. This function is
        nearly identical to __push_facvar_sparse, but instead of pushing
        sparse entries to the constraint matrices, it returns a dense
        vector.
        """
        facvar = [0] * (self.n_vars + 1)
        F = {}
        for i in range(self.matrix_var_dim):
            for j in range(self.matrix_var_dim):
                for key, value in \
                        polynomial[i, j].as_coefficients_dict().items():
                    skey = apply_substitutions(key, self.substitutions,
                                               self.pure_substitution_rules)
                    try:
                        Fk = F[skey]
                    except KeyError:
                        Fk = zeros(self.matrix_var_dim, self.matrix_var_dim)
                    Fk[i, j] += value
                    F[skey] = Fk
        # This is the tracing part
        for key, Fk in F.items():
            if key == S.One:
                k = 1
            else:
                k = self.monomial_index[key]
            for i in range(self.matrix_var_dim):
                for j in range(self.matrix_var_dim):
                    sym_matrix = zeros(self.matrix_var_dim,
                                       self.matrix_var_dim)
                    sym_matrix[i, j] = 1
                    facvar[k+i*self.matrix_var_dim+j] = (sym_matrix*Fk).trace()
        facvar = [float(f) for f in facvar]
        return facvar