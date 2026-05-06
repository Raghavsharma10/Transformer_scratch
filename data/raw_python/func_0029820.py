def update_config(self, key, value):
        """ Creates or updates db config of the VarDictGroup. Requires bound to db tree. """
        dataset = self._top._config.dataset
        session = object_session(self._top._config)
        logger.debug(
            'Updating VarDictGroup config. dataset: {}, type: {}, key: {}, value: {}'.format(
                dataset, self._top._type, key, value))

        if not self._parent._config:
            self._parent.update_config()

        # create or update group config
        self._config, created = get_or_create(
            session, Config,
            d_vid=dataset.vid, type=self._top._type,
            parent=self._parent._config, group=self._key,
            key=self._key,dataset=dataset)
        self._top._add_valid(self._config)

        # create or update value config
        config, created = get_or_create(
            session, Config, parent=self._config, d_vid=dataset.vid,
            type=self._top._type, key=key,dataset=dataset)

        if config.value != value:
            # sync db value with term value.
            config.value = value
            session.merge(config)
            session.commit()
            logger.debug(
                'Config bound to the VarDictGroup key updated. config: {}'.format(config))
        self._top._add_valid(config)