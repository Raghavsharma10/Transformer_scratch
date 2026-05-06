def referencesGenerator(self, request):
        """
        Returns a generator over the (reference, nextPageToken) pairs
        defined by the specified request.
        """
        referenceSet = self.getDataRepository().getReferenceSet(
            request.reference_set_id)
        results = []
        for obj in referenceSet.getReferences():
            include = True
            if request.md5checksum:
                if request.md5checksum != obj.getMd5Checksum():
                    include = False
            if request.accession:
                if request.accession not in obj.getSourceAccessions():
                    include = False
            if include:
                results.append(obj)
        return self._objectListGenerator(request, results)