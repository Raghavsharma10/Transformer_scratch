def validate_query(self, query):
        """Validate a query.

        Determines whether `query` is well-formed. This includes checking for all
        required parameters, as well as checking parameters for valid values.

        Parameters
        ----------
        query : NCSSQuery
            The query to validate

        Returns
        -------
        valid : bool
            Whether `query` is valid.

        """
        # Make sure all variables are in the dataset
        return bool(query.var) and all(var in self.variables for var in query.var)