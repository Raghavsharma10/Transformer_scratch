def _resolve_value(self, name):
        """ Returns an appropriate value for the given name. 
            This simply asks each of the instances for a value.
        """
        for instance in self.__instances():
            value = instance._resolve_value(name)
            if value:
                return value

        # Otherwise, return an appropriate default value (populate_from)
        # TODO: This is duplicated in meta_models. Move this to a common home.
        if name in self.__metadata._meta.elements:
            populate_from = self.__metadata._meta.elements[name].populate_from
            if callable(populate_from):
                return populate_from(None)
            elif isinstance(populate_from, Literal):
                return populate_from.value
            elif populate_from is not NotSet:
                return self._resolve_value(populate_from)