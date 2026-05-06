def load(self, d3mds):
        """Load X, y and context from D3MDS."""
        X, y = d3mds.get_data()

        resource_columns = d3mds.get_related_resources(self.data_modality)
        for resource_column in resource_columns:
            X = self.load_resources(X, resource_column, d3mds)

        context = self.get_context(X, y)

        return Dataset(d3mds.dataset_id, X, y, context=context)