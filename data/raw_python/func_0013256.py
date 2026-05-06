def _gaFeatureForFeatureDbRecord(self, feature):
        """
        :param feature: The DB Row representing a feature
        :return: the corresponding GA4GH protocol.Feature object
        """
        gaFeature = protocol.Feature()
        gaFeature.id = self.getCompoundIdForFeatureId(feature['id'])
        if feature.get('parent_id'):
            gaFeature.parent_id = self.getCompoundIdForFeatureId(
                    feature['parent_id'])
        else:
            gaFeature.parent_id = ""
        gaFeature.feature_set_id = self.getId()
        gaFeature.reference_name = pb.string(feature.get('reference_name'))
        gaFeature.start = pb.int(feature.get('start'))
        gaFeature.end = pb.int(feature.get('end'))
        gaFeature.name = pb.string(feature.get('name'))
        if feature.get('strand', '') == '-':
            gaFeature.strand = protocol.NEG_STRAND
        else:
            # default to positive strand
            gaFeature.strand = protocol.POS_STRAND
        gaFeature.child_ids.extend(map(
                self.getCompoundIdForFeatureId,
                json.loads(feature['child_ids'])))
        gaFeature.feature_type.CopyFrom(
            self._ontology.getGaTermByName(feature['type']))
        attributes = json.loads(feature['attributes'])
        # TODO: Identify which values are ExternalIdentifiers and OntologyTerms
        for key in attributes:
            for v in attributes[key]:
                gaFeature.attributes.attr[key].values.add().string_value = v
        if 'gene_name' in attributes and len(attributes['gene_name']) > 0:
            gaFeature.gene_symbol = pb.string(attributes['gene_name'][0])
        return gaFeature