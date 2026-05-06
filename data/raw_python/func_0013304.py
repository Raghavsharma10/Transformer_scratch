def generateVariantAnnotation(self, variant):
        """
        Generate a random variant annotation based on a given variant.
        This generator should be seeded with a value that is unique to the
        variant so that the same annotation will always be produced regardless
        of the order it is generated in.
        """
        # To make this reproducible, make a seed based on this
        # specific variant.
        seed = self._randomSeed + variant.start + variant.end
        randomNumberGenerator = random.Random()
        randomNumberGenerator.seed(seed)
        ann = protocol.VariantAnnotation()
        ann.variant_annotation_set_id = str(self.getCompoundId())
        ann.variant_id = variant.id
        ann.created = datetime.datetime.now().isoformat() + "Z"
        # make a transcript effect for each alternate base element
        # multiplied by a random integer (1,5)
        for base in variant.alternate_bases:
            ann.transcript_effects.add().CopyFrom(
                self.generateTranscriptEffect(
                    variant, ann, base, randomNumberGenerator))
        ann.id = self.getVariantAnnotationId(variant, ann)
        return ann