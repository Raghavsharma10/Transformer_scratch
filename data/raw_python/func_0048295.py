def _compute_one_closed_central_moment(self, moment, covariance_matrix):
        r"""
        Compute each row of closed central moment based on Isserlis' Theorem of calculating higher order moments
        of multivariate normal distribution in terms of covariance matrix

        :param moment: moment matrix
        :param covariance_matrix: matrix containing variances and covariances
        :return: each row of closed central moment
        """

        # If moment order is odd, higher order moments equals 0
        if moment.order % 2 != 0:
            return sp.Integer(0)

        # index of species
        idx = [i for i in range(len(moment.n_vector))]

        # repeat the index of a species as many time as its value in counter
        list_for_partition = reduce(operator.add, map(lambda i, c: [i] * c, idx, moment.n_vector))

        # If moment order is even, :math: '\mathbb{E} [x_1x_2 \ldots  x_2_n] = \sum \prod\mathbb{E} [x_ix_j] '
        # e.g.:math: '\mathbb{E} [x_1x_2x_3x_4] = \mathbb{E} [x_1x_2] +\mathbb{E} [x_1x_3] +\mathbb{E} [x_1x_4]
        # +\mathbb{E} [x_2x_3]+\mathbb{E} [x_2x_4]+\mathbb{E} [x_3x_4]'
        # For second order moment, there is only one way of partitioning. Hence, no need to generate partitions
        if moment.order == 2:
            return covariance_matrix[list_for_partition[0], list_for_partition[1]]

        # For even moment order other than 2, generate a list of partitions of the indices of covariances
        else:
            each_row = []
            for idx_pair in self._generate_partitions(list_for_partition):
                # Retrieve the pairs of covariances using the pairs of partitioned indices
                l = [covariance_matrix[i, j] for i,j in idx_pair]
                # Calculate the product of each pair of covariances
                each_row.append(product(l))

            # The corresponding closed central moment of that moment order is the sum of the products
            return sum(each_row)