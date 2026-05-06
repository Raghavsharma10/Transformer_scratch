def name(self):
        """Name of the model type.

        For base models, |Model.name| corresponds to the package name:

        >>> from hydpy import prepare_model
        >>> hland = prepare_model('hland')
        >>> hland.name
        'hland'

        For application models, |Model.name| corresponds the module name:

        >>> hland_v1 = prepare_model('hland_v1')
        >>> hland_v1.name
        'hland_v1'

        This last example has only technical reasons:

        >>> hland.name
        'hland'
        """
        name = self.__name
        if name:
            return name
        subs = self.__module__.split('.')
        if len(subs) == 2:
            type(self).__name = subs[1]
        else:
            type(self).__name = subs[2]
        return self.__name