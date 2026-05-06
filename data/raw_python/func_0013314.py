def convertVariantAnnotation(self, record):
        """
        Converts the specfied pysam variant record into a GA4GH variant
        annotation object using the specified function to convert the
        transcripts.
        """
        variant = self._variantSet.convertVariant(record, [])
        annotation = self._createGaVariantAnnotation()
        annotation.variant_id = variant.id
        gDots = record.info.get(b'HGVS.g')
        # Convert annotations from INFO field into TranscriptEffect
        transcriptEffects = []
        annotations = record.info.get(b'ANN') or record.info.get(b'CSQ')
        for i, ann in enumerate(annotations):
            hgvsG = gDots[i % len(variant.alternate_bases)] if gDots else None
            transcriptEffects.append(self.convertTranscriptEffect(ann, hgvsG))
        annotation.transcript_effects.extend(transcriptEffects)
        annotation.id = self.getVariantAnnotationId(variant, annotation)
        return variant, annotation