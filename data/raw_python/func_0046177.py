def _gamma_factorial(self, expr, n):
        r"""
        Compute :math:`\frac {(\alpha)_m = (\alpha + m - 1)!}{(\alpha - 1)!}`
        See Eq. 3 in Gamma moment closure Lakatos 2014 unpublished

        :param expr: a symbolic expression
        :type expr:
        :param n:
        :type n: `int`

        :return: a symbolic expression
        """
        if n == 0:
            return 1
        return product([expr+i for i in range(n)])