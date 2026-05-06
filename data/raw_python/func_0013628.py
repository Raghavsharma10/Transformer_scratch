def getAssociations(
            self, request=None, featureSets=[]):
        """
        This query is the main search mechanism.
        It queries the graph for annotations that match the
        AND of [feature,environment,phenotype].
        """
        if len(featureSets) == 0:
            featureSets = self.getParentContainer().getFeatureSets()
        # query to do search
        query = self._formatFilterQuery(request, featureSets)
        associations = self._rdfGraph.query(query)
        # associations is now a dict with rdflib terms with variable and
        # URIrefs or literals

        # given get the details for the feature,phenotype and environment
        associations_details = self._detailTuples(
            self._extractAssociationsDetails(
                associations))

        # association_details is now a list of {subject,predicate,object}
        # for each of the association detail
        # http://nmrml.org/cv/v1.0.rc1/doc/doc/objectproperties/BFO0000159___-324347567.html
        # label "has quality at all times" (en)
        associationList = []
        for assoc in associations.bindings:
            if '?feature' in assoc:
                association = self._bindingsToDict(assoc)
                association['feature'] = self._getDetails(
                    association['feature'],
                    associations_details)
                association['environment'] = self._getDetails(
                    association['environment'],
                    associations_details)
                association['phenotype'] = self._getDetails(
                    association['phenotype'],
                    associations_details)
                association['evidence'] = association['phenotype'][HAS_QUALITY]
                association['id'] = association['association']
                associationList.append(association)

                # our association list is now a list of dicts with the
                # elements of an association: environment, evidence,
                # feature, phenotype and sources, each with their
                # references (labels or URIrefs)

        # create GA4GH objects
        associations = [
            self._toGA4GH(assoc, featureSets) for
            assoc in associationList]
        return associations