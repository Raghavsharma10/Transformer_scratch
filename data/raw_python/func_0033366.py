def get_cube(self, cube, init=True, name=None, copy_config=True, **kwargs):
        '''wrapper for :func:`metrique.utils.get_cube`

        Locates and loads a metrique cube

        :param cube: name of cube to load
        :param init: (bool) initialize cube before returning?
        :param name: override the name of the cube
        :param copy_config: apply config of calling cube to new?
                            Implies init=True.
        :param kwargs: additional :func:`metrique.utils.get_cube`
        '''
        name = name or cube
        config = copy(self.config) if copy_config else {}
        config_file = self.config_file
        container = type(self.container)
        container_config = copy(self.container_config)
        proxy = str(type(self.proxy))
        return get_cube(cube=cube, init=init, name=name, config=config,
                        config_file=config_file, container=container,
                        container_config=container_config,
                        proxy=proxy, proxy_config=self.proxy_config, **kwargs)