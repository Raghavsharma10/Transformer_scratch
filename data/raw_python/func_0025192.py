def extract_attribute_grid(self, model_grid, potential=False, future=False):
        """
        Extracts the data from a ModelOutput or ModelGrid object within the bounding box region of the STObject.
        
        Args:
            model_grid: A ModelGrid or ModelOutput Object
            potential: Extracts from the time before instead of the same time as the object
        """

        if potential:
            var_name = model_grid.variable + "-potential"
            timesteps = np.arange(self.start_time - 1, self.end_time)
        elif future:
            var_name = model_grid.variable + "-future"
            timesteps = np.arange(self.start_time + 1, self.end_time + 2)
        else:
            var_name = model_grid.variable
            timesteps = np.arange(self.start_time, self.end_time + 1)
        self.attributes[var_name] = []
        for ti, t in enumerate(timesteps):
            self.attributes[var_name].append(
                model_grid.data[t - model_grid.start_hour, self.i[ti], self.j[ti]])