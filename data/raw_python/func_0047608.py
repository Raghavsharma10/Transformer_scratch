def _append_zeros(self, initial_conditions, number_of_equations):
        """If not all intial conditions specified, append zeros to them
           TODO: is this really the best way to do this?
        """

        if len(initial_conditions) < number_of_equations:
            initial_conditions = np.concatenate((initial_conditions,
                                                 [0.0] * (self.problem.number_of_equations - len(initial_conditions))))
        return initial_conditions