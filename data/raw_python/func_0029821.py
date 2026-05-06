def update_config(self):
        """ Creates or updates db config of the term. Requires bound to db tree. """
        dataset = self._top._config.dataset
        session = object_session(self._top._config)

        #logger.debug('Updating term config. dataset: {}, type: {}, key: {}, value: {}'.format(
        #        dataset, self._top._type, self._key, self.get()))

        if not self._parent._config:
            self._parent.update_config()

        self._config, created = _get_config_instance(
            self, session,
            parent=self._parent._config, d_vid=dataset.vid,
            type=self._top._type, key=self._key, dataset=dataset)
        if created:
            self._top._cached_configs[self._get_path()] = self._config

        # We update ScalarTerm and ListTerm values only. Composite terms (DictTerm for example)
        # should not contain value.
        if isinstance(self, (ScalarTerm, ListTerm)):
            if self._config.value != self.get():
                self._config.value = self.get()
                session.merge(self._config)
                session.commit()
        self._top._add_valid(self._config)