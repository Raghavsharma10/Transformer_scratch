def get_contingency_tables(self):
        """
        Create an Array of ContingencyTable objects for each probability threshold.

        Returns:
            Array of ContingencyTable objects
        """
        return np.array([ContingencyTable(*ct) for ct in self.contingency_tables.values])