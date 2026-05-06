def _extractAssociationsDetails(self, associations):
        """
        Given a set of results from our search query, return the
        `details` (feature,environment,phenotype)
        """
        detailedURIRef = []
        for row in associations.bindings:
            if 'feature' in row:
                detailedURIRef.append(row['feature'])
                detailedURIRef.append(row['environment'])
                detailedURIRef.append(row['phenotype'])
        return detailedURIRef