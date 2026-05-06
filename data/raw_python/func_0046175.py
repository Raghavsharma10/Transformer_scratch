def _get_parameter_symbols(self, n_counter, k_counter):
        r"""
        Calculates parameters Y expressions and beta coefficients in
        :math:`X = {A(\beta_0,\beta_1\ldots \beta_n) \cdot Y}`

        :param n_counter: a list of :class:`~means.core.descriptors.Moment`\s representing central moments
        :type n_counter: list[:class:`~means.core.descriptors.Moment`]
        :param k_counter: a list of :class:`~means.core.descriptors.Moment`\s representing raw moments
        :type k_counter: list[:class:`~means.core.descriptors.Moment`]
        :return: two column matrices Y expressions and beta multipliers
        """

        n_moment = self.max_order + 1

        expectation_symbols = sp.Matrix([n.symbol for n in k_counter if n.order == 1])


        n_species = len(expectation_symbols)

        # Create auxiliary symbolic species Y_{ij}, for i,j = 0 ... (n-1) and mirror, so that Y_{ij}=Y_{ji}
        symbolic_species=sp.Matrix([[sp.Symbol(('Y_{0}'.format(str(j)))+'{0}'.format(str(i))) for i in range(n_species)]for j in range(n_species)])
        for i in range(n_species):
            for j in range(i+1,n_species):
                symbolic_species[j,i]=symbolic_species[i,j]


        # Obtain beta terms explaining how original variables are derived from auxiliary ones
        if self.is_multivariate:
            # :math: `X_i = \sum_j Y_{ij}`
            beta_in_matrix = sp.Matrix([sum(symbolic_species[i,:]) for i in range(n_species)])
        else :
            # In univariate case, only the diagonal elements are needed
            beta_in_matrix = sp.Matrix([symbolic_species[i,i] for i in range(n_species)])

        # Covariance symbols are read into a matrix. Variances are the diagonal elements
        covariance_matrix = sp.Matrix(n_species,n_species, lambda x,y: self._get_covariance_symbol(n_counter,x,y))
        variance_symbols = sp.Matrix([covariance_matrix[i,i] for i in range(n_species)])


        # Compute :math:  `\beta_i = Var(X_i)/\mathbb{E}(X_i) \bar\alpha_i = \mathbb{E}(X_i)^2/Var(X_i)`
        beta_exprs = sp.Matrix([v / e for e,v in zip(expectation_symbols,variance_symbols)])
        alpha_bar_exprs = sp.Matrix([(e ** 2) / v for e,v in zip(expectation_symbols,variance_symbols)])

        if self.is_multivariate:
            # Calculate nondiagonal elements from covariances
            alpha_exprs = sp.Matrix(n_species,n_species, lambda i,j: covariance_matrix[i,j]/(beta_exprs[i]*beta_exprs[j]))
        else:
            # Covariances are zero in univariate case
            alpha_exprs = sp.Matrix(n_species,n_species, lambda i,j: 0)
        for sp_idx in range(n_species):
            # Compute diagonal elements as :math: `\alpha_{ii} = \bar\alpha_{i} - \sum(\alpha_{ij})` //equiv to \bar\alpha_{i} in univariate
            alpha_exprs[sp_idx,sp_idx]=0
            alpha_exprs[sp_idx,sp_idx]=alpha_bar_exprs[sp_idx]-sum(alpha_exprs[sp_idx,:])


        # Each row in moment matrix contains the exponents of Xs for a given moment
        # Each row in Y_exprs and beta_multipliers has elements on the appropriate power
        # determined by the corresponding row in the moment matrix
        Y_exprs = []
        beta_multipliers = []

        positive_n_counter = [n for n in n_counter if n.order > 0]
        for mom in positive_n_counter:
            Y_exprs.append(product([(b ** s).expand() for b, s in zip(beta_in_matrix, mom.n_vector)]))
            beta_multipliers.append(product([(b ** s).expand() for b, s in zip(beta_exprs, mom.n_vector)]))

        Y_exprs = sp.Matrix(Y_exprs).applyfunc(sp.expand)
        beta_multipliers = sp.Matrix(beta_multipliers)

        # Substitute alpha expressions in place of symbolic species Ys
        # by going through all powers up to the moment order for closure
        subs_pairs = []
        for i,a in enumerate(alpha_exprs):
            Y_to_substitute = [symbolic_species[i]**n for n in range(2, n_moment+1)]

            # Obtain alpha term for higher order moments :math: `\mathbb{E}(Y_{ij}^n) \rightarrow (\alpha_{ij})_n`
            alpha_m = [self._gamma_factorial(a,n) for n in range(2, n_moment+1)]

            # Substitute alpha term for symbolic species
            subs_pairs += zip(Y_to_substitute, alpha_m)
            subs_pairs.append((symbolic_species[i], a)) # Add first order expression to the end
        Y_exprs = substitute_all(Y_exprs, subs_pairs)

        return Y_exprs, beta_multipliers