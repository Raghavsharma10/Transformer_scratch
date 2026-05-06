def to_query(self):
        """
        Returns a json-serializable representation.
        """
        return {
            self.name: {
                'lang': self.lang,
                'script': self.script,
                'params': self.script_params
            }
        }