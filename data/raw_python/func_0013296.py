def convertVariant(self, record, callSetIds):
        """
        Converts the specified pysam variant record into a GA4GH Variant
        object. Only calls for the specified list of callSetIds will
        be included.
        """
        variant = self._createGaVariant()
        variant.reference_name = record.contig
        if record.id is not None:
            variant.names.extend(record.id.split(';'))
        variant.start = record.start          # 0-based inclusive
        variant.end = record.stop             # 0-based exclusive
        variant.reference_bases = record.ref
        if record.alts is not None:
            variant.alternate_bases.extend(list(record.alts))
        filterKeys = record.filter.keys()
        if len(filterKeys) == 0:
            variant.filters_applied = False
        else:
            variant.filters_applied = True
            if len(filterKeys) == 1 and filterKeys[0] == 'PASS':
                variant.filters_passed = True
            else:
                variant.filters_passed = False
                variant.filters_failed.extend(filterKeys)
        # record.qual is also available, when supported by GAVariant.
        for key, value in record.info.iteritems():
            if value is None:
                continue
            if key == 'SVTYPE':
                variant.variant_type = value
            elif key == 'SVLEN':
                variant.svlen = int(value[0])
            elif key == 'CIPOS':
                variant.cipos.extend(value)
            elif key == 'CIEND':
                variant.ciend.extend(value)
            elif isinstance(value, str):
                value = value.split(',')
            protocol.setAttribute(
                variant.attributes.attr[key].values, value)
        for callSetId in callSetIds:
            callSet = self.getCallSet(callSetId)
            pysamCall = record.samples[str(callSet.getSampleName())]
            variant.calls.add().CopyFrom(
                self._convertGaCall(callSet, pysamCall))
        variant.id = self.getVariantId(variant)
        return variant