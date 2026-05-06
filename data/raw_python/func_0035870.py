def _getParentClass(self, startClass, parentClass):
        """ gets the parent class by calling successive parent classes with .parent until parentclass is matched.
        """
        try:
            if not startClass:  # reached system with no hits
                raise AttributeError
        except AttributeError:  # i.e calling binary on an object without one
                raise HierarchyError('This object ({0}) has no {1} as a parent object'.format(self.name, parentClass))

        if startClass.classType == parentClass:
            return startClass
        else:
            return self._getParentClass(startClass.parent, parentClass)