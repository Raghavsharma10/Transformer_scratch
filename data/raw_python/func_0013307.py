def getVariantAnnotations(self, referenceName, startPosition, endPosition):
        """
        Generator for iterating through variant annotations in this
        variant annotation set.
        :param referenceName:
        :param startPosition:
        :param endPosition:
        :return: generator of protocol.VariantAnnotation
        """
        variantIter = self._variantSet.getPysamVariants(
            referenceName, startPosition, endPosition)
        for record in variantIter:
            yield self.convertVariantAnnotation(record)