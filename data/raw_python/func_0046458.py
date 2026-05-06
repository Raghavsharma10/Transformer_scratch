def _compute_raw_moments(self, n_counter, k_counter):

        # The symbols for expectations are simply the first order raw moments.
        """
        :param n_counter: a list of :class:`~means.core.descriptors.Moment`\s representing central moments
        :type n_counter: list[:class:`~means.core.descriptors.Moment`]
        :param k_counter: a list of :class:`~means.core.descriptors.Moment`\s representing raw moments
        :type k_counter: list[:class:`~means.core.descriptors.Moment`]
        :return: a vector of parametric expression for raw moments
        """
        expectation_symbols = [pm.symbol for pm in k_counter if pm.order == 1]

        n_species = len(expectation_symbols)

        # The covariance expressed in terms of central moment symbols (typically, yxNs, where N is an integer)
        covariance_matrix = sp.Matrix(n_species,n_species, lambda x,y: self._get_covariance_symbol(n_counter,x,y))

        # Variances is the diagonal of covariance matrix
        variance_symbols = [covariance_matrix[i, i] for i in range(n_species)]

        # :math: '\logVar(x_i) = 1 + \frac { Var(x_i)}{ \mathbb{E} (x_i)^2}'
        log_variance_symbols = sp.Matrix([sp.log(sp.Integer(1) + v/(e ** sp.Integer(2))) for e,v
                                          in zip(expectation_symbols, variance_symbols)])

        # :math: '\log\mathbb{E} (x_i) = \log(\mathbb{E} (x_i) )+ \frac {\log (Var(x_i))}{2}'
        log_expectation_symbols = sp.Matrix([sp.log(e) - lv/sp.Integer(2) for e,lv
                                             in zip(expectation_symbols, log_variance_symbols)])

        # Assign log variance symbols on the diagonal of size n_species by n_species
        log_variance_mat = sp.Matrix(n_species,n_species, lambda x,y: log_variance_symbols[x] if x == y else 0)

        # Assign log covariances and log variances in the matrix log_covariance matrix based on matrix indices
        log_covariance_matrix = sp.Matrix(n_species,n_species, lambda x, y:
                self._get_log_covariance(log_variance_mat, log_expectation_symbols, covariance_matrix, x, y))

        # The n_vectors (e.g. [0,2,0]) of the central moments
        pm_n_vecs = [sp.Matrix(pm.n_vector) for pm in n_counter if pm.order > 1]


        out_mat = sp.Matrix([n.T * (log_covariance_matrix * n) / sp.Integer(2) + n.T * log_expectation_symbols for n in pm_n_vecs])
        # return the exponential of all values

        out_mat = out_mat.applyfunc(lambda x: sp.exp(x))
        return out_mat