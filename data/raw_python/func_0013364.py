def insertReadGroup(self, readGroup):
        """
        Inserts the specified readGroup into the DB.
        """
        statsJson = json.dumps(protocol.toJsonDict(readGroup.getStats()))
        experimentJson = json.dumps(
            protocol.toJsonDict(readGroup.getExperiment()))
        try:
            models.Readgroup.create(
                id=readGroup.getId(),
                readgroupsetid=readGroup.getParentContainer().getId(),
                name=readGroup.getLocalId(),
                predictedinsertedsize=readGroup.getPredictedInsertSize(),
                samplename=readGroup.getSampleName(),
                description=readGroup.getDescription(),
                stats=statsJson,
                experiment=experimentJson,
                biosampleid=readGroup.getBiosampleId(),
                attributes=json.dumps(readGroup.getAttributes()))
        except Exception as e:
            raise exceptions.RepoManagerException(e)