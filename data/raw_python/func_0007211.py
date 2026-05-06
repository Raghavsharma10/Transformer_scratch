def instance(self, *args, **kwargs):
        """Create an instance of the specified class in the initializer.

        :param args: the arguments given to the initializer of the new class
        :param kwargs: the keyword arguments given to the initializer of the
                     new class

        """
        mod, cls = self.get_module_class()
        inst = cls(*args, **kwargs)
        logger.debug(f'inst: {inst}')
        return inst