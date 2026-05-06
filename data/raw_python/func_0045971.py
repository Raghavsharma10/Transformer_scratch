def load(self, d3mds):
        """Load X, y and context from D3MDS."""
        X, y = d3mds.get_data()

        return Dataset(d3mds.dataset_id, X, y)