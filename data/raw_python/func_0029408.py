def fill_null_values(self):
        """ Fill missing model fields in JSON with {key: null value}.

        Only run for PUT requests.
        """
        if not self.Model:
            log.info("%s has no model defined" % self.__class__.__name__)
            return

        empty_values = self.Model.get_null_values()
        for field, value in empty_values.items():
            if field not in self._json_params:
                self._json_params[field] = value