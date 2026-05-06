def referenceSetsGenerator(self, request):
        """
        Returns a generator over the (referenceSet, nextPageToken) pairs
        defined by the specified request.
        """
        results = []
        for obj in self.getDataRepository().getReferenceSets():
            include = True
            if request.md5checksum:
                if request.md5checksum != obj.getMd5Checksum():
                    include = False
            if request.accession:
                if request.accession not in obj.getSourceAccessions():
                    include = False
            if request.assembly_id:
                if request.assembly_id != obj.getAssemblyId():
                    include = False
            if include:
                results.append(obj)
        return self._objectListGenerator(request, results)