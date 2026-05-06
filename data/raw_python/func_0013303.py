def getVariantAnnotationId(self, gaVariant, gaAnnotation):
        """
        Produces a stringified compoundId representing a variant
        annotation.
        :param gaVariant:   protocol.Variant
        :param gaAnnotation: protocol.VariantAnnotation
        :return:  compoundId String
        """
        md5 = self.hashVariantAnnotation(gaVariant, gaAnnotation)
        compoundId = datamodel.VariantAnnotationCompoundId(
            self.getCompoundId(), gaVariant.reference_name,
            str(gaVariant.start), md5)
        return str(compoundId)