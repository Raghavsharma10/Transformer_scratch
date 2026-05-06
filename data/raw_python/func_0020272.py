def unregister(self, model=None):
        '''Unregister a ``model`` if provided, otherwise it unregister all
registered models. Return a list of unregistered model managers or ``None``
if no managers were removed.'''
        if model is not None:
            try:
                manager = self._registered_models.pop(model)
            except KeyError:
                return
            if self._registered_names.get(manager._meta.name) == manager:
                self._registered_names.pop(manager._meta.name)
            return [manager]
        else:
            managers = list(self._registered_models.values())
            self._registered_models.clear()
            return managers