def _getFeatureById(self, featureId):
        """
        find a feature and return ga4gh representation, use 'native' id as
        featureId
        """
        featureRef = rdflib.URIRef(featureId)
        featureDetails = self._detailTuples([featureRef])
        feature = {}
        for detail in featureDetails:
            feature[detail['predicate']] = []

        for detail in featureDetails:
            feature[detail['predicate']].append(detail['object'])

        pbFeature = protocol.Feature()

        term = protocol.OntologyTerm()
        # Schema for feature only supports one type of `type`
        # here we default to first OBO defined
        for featureType in sorted(feature[TYPE]):
            if "obolibrary" in featureType:
                term.term = self._featureTypeLabel(featureType)
                term.term_id = featureType
                pbFeature.feature_type.MergeFrom(term)
                break

        pbFeature.id = featureId
        # Schema for feature only supports one type of `name` `symbol`
        # here we default to shortest for symbol and longest for name
        feature[LABEL].sort(key=len)
        pbFeature.gene_symbol = feature[LABEL][0]
        pbFeature.name = feature[LABEL][-1]

        pbFeature.attributes.MergeFrom(protocol.Attributes())
        for key in feature:
            for val in sorted(feature[key]):
                pbFeature.attributes.attr[key].values.add().string_value = val

        if featureId in self._locationMap:
            location = self._locationMap[featureId]
            pbFeature.reference_name = location["chromosome"]
            pbFeature.start = location["begin"]
            pbFeature.end = location["end"]

        return pbFeature