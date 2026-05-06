def _update_dict(self, to_dict, from_dict):
        """ Recursively merges the fields for two dictionaries.

        Args:
            to_dict (dict): The dictionary onto which the merge is executed.
            from_dict (dict): The dictionary merged into to_dict
        """
        for key, value in from_dict.items():
            if key in to_dict and isinstance(to_dict[key], dict) and \
                    isinstance(from_dict[key], dict):
                self._update_dict(to_dict[key], from_dict[key])
            else:
                to_dict[key] = from_dict[key]