def extract_attribute_array(self, data_array, var_name):
        """
        Extracts data from a 2D array that has the same dimensions as the grid used to identify the object.

        Args:
            data_array: 2D numpy array

        """
        if var_name not in self.attributes.keys():
            self.attributes[var_name] = []
        for t in range(self.times.size):
            self.attributes[var_name].append(data_array[self.i[t], self.j[t]])