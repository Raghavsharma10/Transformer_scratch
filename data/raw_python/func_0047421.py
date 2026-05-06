def _generate_mass_fluctuation_kinetics(self, central_moments, dmu_over_dt, n_counter):
        """
        Generate the Mass Fluctuation Kinetics (i.e. the right hand side of the ODEs)

        :param central_moments: The matrix of central moment expressions
        :param dmu_over_dt:
        :param n_counter: a list of :class:`~means.core.descriptors.Moment`\s representing central moments
        :type n_counter: list[:class:`~means.core.descriptors.Moment`]
        :return: the MFK as a matrix
        :rtype: :class:`sympy.Matrix`
        """

        # symbols for central moments
        central_moments_symbols = sp.Matrix([n.symbol for n in n_counter])

        # rhs for the first order raw moment
        mfk = [e for e in dmu_over_dt * central_moments_symbols]
        # rhs for the higher order raw moments
        mfk += [(sp.Matrix(cm).T * central_moments_symbols)[0] for cm in central_moments.tolist()]

        mfk = sp.Matrix(mfk)

        return mfk