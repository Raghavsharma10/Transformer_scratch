def get_subassistants(self):
        """Return list of instantiated subassistants.

        Usually, this needs not be overriden in subclasses, you should just override
        get_subassistant_classes

        Returns:
            list of instantiated subassistants
        """
        if not hasattr(self, '_subassistants'):
            self._subassistants = []
            # we want to know, if type(self) defines 'get_subassistant_classes',
            # we don't want to inherit it from superclass (would cause recursion)
            if 'get_subassistant_classes' in vars(type(self)):
                for a in self.get_subassistant_classes():
                    self._subassistants.append(a())
        return self._subassistants