def fetch_data(self, **var):
        """Retrieve data from CDMRemote for one or more variables."""
        varstr = ','.join(name + self._convert_indices(ind)
                          for name, ind in var.items())
        query = self.query().add_query_parameter(req='data', var=varstr)
        return self._fetch(query)