def _createGaVariant(self):
        """
        Convenience method to set the common fields in a GA Variant
        object from this variant set.
        """
        ret = protocol.Variant()
        if self._creationTime:
            ret.created = self._creationTime
        if self._updatedTime:
            ret.updated = self._updatedTime
        ret.variant_set_id = self.getId()
        return ret