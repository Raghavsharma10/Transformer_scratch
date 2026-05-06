def getVariantSetByName(self, name):
        """
        Returns a VariantSet with the specified name, or raises a
        VariantSetNameNotFoundException if it does not exist.
        """
        if name not in self._variantSetNameMap:
            raise exceptions.VariantSetNameNotFoundException(name)
        return self._variantSetNameMap[name]