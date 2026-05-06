def _get_log_covariance(self, log_variance_mat, log_expectation_symbols, covariance_matrix, x, y):
        r"""
        Compute log covariances according to:\\

        :math:`\log{(Cov(x_i,x_j))} = \frac { 1 + Cov(x_i,x_j)}{\exp[\log \mathbb{E}(x_i) + \log \mathbb{E}(x_j)+\frac{1}{2} (\log Var(x_i) + \log Var(x_j)]}`

        :param log_variance_mat: a column matrix of log variance
        :param log_expectation_symbols: a column matrix of log expectations
        :param covariance_matrix: a matrix of covariances
        :param x: x-coordinate in matrix of log variances and log covariances
        :param y: y-coordinate in matrix of log variances and log covariances
        :return: the log covariance between x and y
        """
        # The diagonal of the return matrix includes all the log variances
        if x == y:
            return log_variance_mat[x, x]
        # log covariances are calculated if not on the diagonal of the return matrix
        elif self.is_multivariate:
            denom = sp.exp(log_expectation_symbols[x] +
                           log_expectation_symbols[y] +
                           (log_variance_mat[x, x] + log_variance_mat[y, y])/ sp.Integer(2))
            return sp.log(sp.Integer(1) + covariance_matrix[x, y] / denom)
        # univariate case: log covariances are 0s.
        else:
            return sp.Integer(0)