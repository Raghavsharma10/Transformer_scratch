def right_hand_side_as_function(self):
        """
        Generates and returns the right hand side of the model as a callable function that takes two parameters:
        values for variables and values for constants,
        e.g. `f(values_for_variables=[1,2,3], values_for_constants=[3,4,5])

        This function is directly used in `means.simulation.Simulation`
        :return:
        :rtype: function
        """
        wrapped_functions = self._right_hand_side_as_numeric_functions

        def f(values_for_variables, values_for_constants):
            all_values = np.concatenate((values_for_constants, values_for_variables))
            ans = np.array([w_f(*all_values) for w_f in wrapped_functions])
            return ans

        return f