def _get_covariance_symbol(self, q_counter, sp1_idx, sp2_idx):
        r"""
        Compute second order moments i.e. variances and covariances
        Covariances equal to 0 in univariate case

        :param q_counter: moment matrix
        :param sp1_idx: index of one species
        :param sp2_idx: index of another species
        :return: second order moments matrix of size n_species by n_species
        """
        # The diagonal positions in the matrix are the variances
        if sp1_idx == sp2_idx:
            return [q.symbol for q in q_counter if q.n_vector[sp1_idx] == 2 and q.order == 2][0]
        # Covariances are found if the moment order is 2 and the moment vector contains double 1
        return [q.symbol for q in q_counter if q.n_vector[sp1_idx] == 1 and q.n_vector[sp2_idx] == 1 and q.order == 2][0]