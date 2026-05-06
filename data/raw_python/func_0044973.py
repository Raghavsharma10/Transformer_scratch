def _make_model(self, data, key=None):
        """
        Creates a model instance with the given data.

        Args:
            data: Model data returned from DB.
            key: Object key
        Returns:
            pyoko.Model object.
        """
        if data['deleted'] and not self.adapter.want_deleted:
            raise ObjectDoesNotExist('Deleted object returned')
        model = self._model_class(self._current_context,
                                  _pass_perm_checks=self._pass_perm_checks)

        model.setattr('key', ub_to_str(key) if key else ub_to_str(data.get('key')))
        model = model.set_data(data, from_db=True)
        model._initial_data = model.clean_value()
        return model