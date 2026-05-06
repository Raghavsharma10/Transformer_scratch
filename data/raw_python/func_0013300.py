def _createGaVariantAnnotation(self):
        """
        Convenience method to set the common fields in a GA VariantAnnotation
        object from this variant set.
        """
        ret = protocol.VariantAnnotation()
        ret.created = self._creationTime
        ret.variant_annotation_set_id = self.getId()
        return ret