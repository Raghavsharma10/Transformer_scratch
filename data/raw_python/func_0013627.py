def _toGA4GH(self, association, featureSets=[]):
        """
        given an association dict,
        return a protocol.FeaturePhenotypeAssociation
        """

        # The association dict has the keys: environment, environment
        # label, evidence, feature label, phenotype and sources. Each
        # key's value is a dict with the RDF predicates as keys and
        # subject as values

        # 1) map a GA4GH FeaturePhenotypeAssociation
        # from the association dict passed to us
        feature = association['feature']

        fpa = protocol.FeaturePhenotypeAssociation()
        fpa.id = association['id']

        feature_id = feature['id']
        for feature_set in featureSets:
            if self.getLocalId() in feature_set.getLocalId():
                feature_id = feature_set.getCompoundIdForFeatureId(feature_id)

        fpa.feature_ids.extend([feature_id])

        msg = 'Association: genotype:[{}] phenotype:[{}] environment:[{}] ' \
              'evidence:[{}] publications:[{}]'
        fpa.description = msg.format(
            association['feature_label'],
            association['phenotype_label'],
            association['environment_label'],
            self._getIdentifier(association['evidence']),
            association['sources']
            )

        # 2) map a GA4GH Evidence
        # from the association's phenotype & evidence
        evidence = protocol.Evidence()
        phenotype = association['phenotype']

        term = protocol.OntologyTerm()
        term.term = association['evidence_type']
        term.term_id = phenotype['id']
        evidence.evidence_type.MergeFrom(term)

        evidence.description = self._getIdentifier(association['evidence'])

        # 3) Store publications from the list of sources
        for source in association['sources'].split("|"):
            evidence.info['publications'].values.add().string_value = source
        fpa.evidence.extend([evidence])

        # 4) map environment (drug) to environmentalContext
        environmentalContext = protocol.EnvironmentalContext()
        environment = association['environment']
        environmentalContext.id = environment['id']
        environmentalContext.description = association['environment_label']

        term = protocol.OntologyTerm()
        term.term = environment['id']
        term.term_id = 'http://purl.obolibrary.org/obo/RO_0002606'
        environmentalContext.environment_type.MergeFrom(term)

        fpa.environmental_contexts.extend([environmentalContext])

        # 5) map the phenotype
        phenotypeInstance = protocol.PhenotypeInstance()
        term = protocol.OntologyTerm()
        term.term = phenotype[TYPE]
        term.term_id = phenotype['id']
        phenotypeInstance.type.MergeFrom(term)

        phenotypeInstance.description = phenotype[LABEL]
        phenotypeInstance.id = phenotype['id']
        fpa.phenotype.MergeFrom(phenotypeInstance)
        fpa.phenotype_association_set_id = self.getId()
        return fpa