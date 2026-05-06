def _clean_pivot_attributes(self, model):
        """
        Get the pivot attributes from a model.

        :type model: eloquent.Model
        """
        values = {}
        delete_keys = []

        for key, value in model.get_attributes().items():
            if key.find('pivot_') == 0:
                values[key[6:]] = value

                delete_keys.append(key)

        for key in delete_keys:
            delattr(model, key)

        return values