def hashVariantAnnotation(cls, gaVariant, gaVariantAnnotation):
        """
        Produces an MD5 hash of the gaVariant and gaVariantAnnotation objects
        """
        treffs = [treff.id for treff in gaVariantAnnotation.transcript_effects]
        return hashlib.md5(
            "{}\t{}\t{}\t".format(
                gaVariant.reference_bases, tuple(gaVariant.alternate_bases),
                treffs)
            ).hexdigest()