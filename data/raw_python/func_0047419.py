def run(self):
        r"""
        Overrides the default run() method.
        Performs the complete analysis on the model specified during initialisation.

        :return: an ODE problem which can be further used in inference and simulation.
        :rtype: :class:`~means.core.problems.ODEProblem`
        """
        max_order = self.__max_order
        stoichiometry_matrix = self.model.stoichiometry_matrix
        propensities = self.model.propensities
        species = self.model.species
        # compute n_counter and k_counter; the "n" and "k" vectors in equations, respectively.
        n_counter, k_counter = generate_n_and_k_counters(max_order, species)
        # dmu_over_dt has row per species and one col per element of n_counter (eq. 6)
        dmu_over_dt = generate_dmu_over_dt(species, propensities, n_counter, stoichiometry_matrix)
        # Calculate expressions to use in central moments equations (eq. 9)
        central_moments_exprs = eq_central_moments(n_counter, k_counter, dmu_over_dt, species, propensities, stoichiometry_matrix, max_order)
        # Expresses central moments in terms of raw moments (and central moments) (eq. 8)
        central_from_raw_exprs = raw_to_central(n_counter, species, k_counter)
        # Substitute raw moment, in central_moments, with expressions depending only on central moments
        central_moments_exprs = self._substitute_raw_with_central(central_moments_exprs, central_from_raw_exprs, n_counter, k_counter)
        # Get final right hand side expressions for each moment in a vector
        mfk = self._generate_mass_fluctuation_kinetics(central_moments_exprs, dmu_over_dt, n_counter)
        # Applies moment expansion closure, that is replaces last order central moments by parametric expressions
        mfk = self.closure.close(mfk, central_from_raw_exprs, n_counter, k_counter)
        # These are the left hand sign symbols referring to the mfk
        prob_lhs = self._generate_problem_left_hand_side(n_counter, k_counter)
        # Finally, we build the problem
        out_problem = ODEProblem("MEA", prob_lhs, mfk, sp.Matrix(self.model.parameters))
        return out_problem