def _arithmetic_operation(self, other, operation):
        """
        Applies an operation between the values of a trajectories and a scalar or between
        the respective values of two trajectories. In the latter case, trajectories should have
        equal descriptions and time points
        """
        if isinstance(other, TrajectoryWithSensitivityData):
            if self.description != other.description:
                raise Exception("Cannot add trajectories with different descriptions")
            if not np.array_equal(self.timepoints, other.timepoints):
                raise Exception("Cannot add trajectories with different time points")
            new_values = operation(self.values, other.values)
            new_sensitivity_data = [operation(ssd, osd) for ssd, osd in
                                    zip(self.sensitivity_data, other.sensitivity_data)]

        elif isinstance(other, numbers.Real):
            new_values = operation(self.values, float(other))
            new_sensitivity_data = [operation(ssd, float(other)) for ssd in self.sensitivity_data]

        else:
            raise Exception("Arithmetic operations is between two `TrajectoryWithSensitivityData`\
                            objects or a `TrajectoryWithSensitivityData` and a scalar.")

        return TrajectoryWithSensitivityData(self.timepoints, new_values, self.description, new_sensitivity_data )