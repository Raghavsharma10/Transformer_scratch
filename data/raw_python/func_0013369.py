def insertReadGroupSet(self, readGroupSet):
        """
        Inserts a the specified readGroupSet into this repository.
        """
        programsJson = json.dumps(
            [protocol.toJsonDict(program) for program in
             readGroupSet.getPrograms()])
        statsJson = json.dumps(protocol.toJsonDict(readGroupSet.getStats()))
        try:
            models.Readgroupset.create(
                id=readGroupSet.getId(),
                datasetid=readGroupSet.getParentContainer().getId(),
                referencesetid=readGroupSet.getReferenceSet().getId(),
                name=readGroupSet.getLocalId(),
                programs=programsJson,
                stats=statsJson,
                dataurl=readGroupSet.getDataUrl(),
                indexfile=readGroupSet.getIndexFile(),
                attributes=json.dumps(readGroupSet.getAttributes()))
            for readGroup in readGroupSet.getReadGroups():
                self.insertReadGroup(readGroup)
        except Exception as e:
            raise exceptions.RepoManagerException(e)