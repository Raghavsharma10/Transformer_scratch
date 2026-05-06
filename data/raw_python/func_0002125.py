def get_manager(self, model):
        """
        Return the active manager for the given model.
        :param model: Model class to look up the manager instance for.
        :return: Manager instance for the model associated with this client.
        """
        if isinstance(model, six.string_types):
            # undocumented string lookup
            for k, m in self._manager_map.items():
                if k.__name__ == model:
                    return m
            else:
                raise KeyError(model)

        return self._manager_map[model]