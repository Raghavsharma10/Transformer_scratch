def extract_tendency_grid(self, model_grid):
        """
        Extracts the difference in model outputs

        Args:
            model_grid: ModelOutput or ModelGrid object.

        """
        var_name = model_grid.variable + "-tendency"
        self.attributes[var_name] = []
        timesteps = np.arange(self.start_time, self.end_time + 1)
        for ti, t in enumerate(timesteps):
            t_index = t - model_grid.start_hour
            self.attributes[var_name].append(
                model_grid.data[t_index, self.i[ti], self.j[ti]] - model_grid.data[t_index - 1, self.i[ti], self.j[ti]]
                )