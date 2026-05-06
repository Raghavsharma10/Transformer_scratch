def clean_query(self):
        """
        Removes any `None` value from an elasticsearch query.
        """
        if self.query:
            for key, value in self.query.items():
                if isinstance(value, list) and None in value:
                    self.query[key] = [v for v in value if v is not None]