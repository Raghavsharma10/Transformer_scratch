def _arithmetic_operation(self, other, operation):
        """
        Applies an operation between the values of a trajectories and a scalar or between
        the respective values of two trajectories. In the latter case, trajectories should have
        equal descriptions and time points
        """
        if isinstance(other, Trajectory):
            if self.description != other.description:
                raise Exception("Cannot add trajectories with different descriptions")
            if not np.array_equal(self.timepoints, other.timepoints):
                raise Exception("Cannot add trajectories with different time points")
            new_values = operation(self.values, other.values)
        elif isinstance(other, numbers.Real):
            new_values = operation(self.values, float(other))
        else:
            raise Exception("Arithmetic operations is between two `Trajectory` objects or a `Trajectory` and a scalar.")

        return Trajectory(self.timepoints, new_values, self.description)