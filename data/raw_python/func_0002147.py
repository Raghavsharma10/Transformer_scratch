def fetch_feature_type(self):
        """Request the featureType from the endpoint."""
        query = self.query().add_query_parameter(req='featureType')
        return self.get_query(query).content