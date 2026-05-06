def instance(self, name=None, *args, **kwargs):
        """Create a new instance using key ``name``.

        :param name: the name of the class (by default) or the key name of the
            class used to find the class
        :param args: given to the ``__init__`` method
        :param kwargs: given to the ``__init__`` method

        """
        logger.info(f'new instance of {name}')
        t0 = time()
        name = self.default_name if name is None else name
        logger.debug(f'creating instance of {name}')
        class_name, params = self._class_name_params(name)
        cls = self._find_class(class_name)
        params.update(kwargs)
        if self._has_init_config(cls):
            logger.debug(f'found config parameter')
            params['config'] = self.config
        if self._has_init_name(cls):
            logger.debug(f'found name parameter')
            params['name'] = name
        if logger.level >= logging.DEBUG:
            for k, v in params.items():
                logger.debug(f'populating {k} -> {v} ({type(v)})')
        inst = self._instance(cls, *args, **params)
        logger.info(f'created {name} instance of {cls.__name__} ' +
                    f'in {(time() - t0):.2f}s')
        return inst